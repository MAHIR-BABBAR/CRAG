"""SciFact preparation and cache reuse, driven by a stand-in for ir_datasets."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from custom_rag.eval.datasets import prepare_scifact


@dataclass(frozen=True)
class _Doc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class _Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class _Qrel:
    query_id: str
    doc_id: str
    relevance: int


class _Dataset:
    """Two judged queries, one unjudged query, and filler documents."""

    def __init__(self) -> None:
        self.load_count = 0

    def docs_iter(self):
        yield _Doc("doc-a", "Alpha", "Alpha body text.")
        yield _Doc("doc-b", "Beta", "Beta body text.")
        for i in range(8):
            yield _Doc(f"filler-{i}", "", f"Filler body {i}.")
        yield _Doc("doc-empty", "", "   ")

    def queries_iter(self):
        yield _Query("q1", "what is alpha")
        yield _Query("q2", "what is beta")
        yield _Query("q3", "unjudged question")

    def qrels_iter(self):
        yield _Qrel("q1", "doc-a", 1)
        yield _Qrel("q2", "doc-b", 1)
        yield _Qrel("q2", "doc-missing", 0)


@pytest.fixture
def fake_ir_datasets(monkeypatch: pytest.MonkeyPatch) -> _Dataset:
    dataset = _Dataset()
    module = types.ModuleType("ir_datasets")

    def load(name: str) -> _Dataset:
        dataset.load_count += 1
        return dataset

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ir_datasets", module)
    return dataset


def test_prepare_writes_docs_qrels_and_manifest(tmp_path: Path, fake_ir_datasets: _Dataset) -> None:
    prepared = prepare_scifact(tmp_path / "scifact")

    assert prepared.doc_count == 10
    assert prepared.query_count == 2  # the unjudged query is dropped
    assert prepared.fingerprint
    assert prepared.subset is False

    qrels = json.loads(prepared.qrels_path.read_text(encoding="utf-8"))
    ids = {entry["id"] for entry in qrels["queries"]}
    assert ids == {"q1", "q2"}
    # A qrel with relevance 0 is not a positive judgement.
    q2 = next(entry for entry in qrels["queries"] if entry["id"] == "q2")
    assert q2["relevant"] == [{"external_id": "doc-b"}]


def test_documents_carry_their_external_id(tmp_path: Path, fake_ir_datasets: _Dataset) -> None:
    prepared = prepare_scifact(tmp_path / "scifact")
    sidecar = prepared.docs_dir / "doc-a.meta.json"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"external_id": "doc-a"}
    assert "Alpha" in (prepared.docs_dir / "doc-a.txt").read_text(encoding="utf-8")


def test_blank_documents_are_not_written(tmp_path: Path, fake_ir_datasets: _Dataset) -> None:
    prepared = prepare_scifact(tmp_path / "scifact")
    assert not (prepared.docs_dir / "doc-empty.txt").exists()


def test_matching_cache_is_reused(tmp_path: Path, fake_ir_datasets: _Dataset) -> None:
    root = tmp_path / "scifact"
    prepare_scifact(root)
    prepare_scifact(root)
    assert fake_ir_datasets.load_count == 1


def test_force_rebuilds_the_cache(tmp_path: Path, fake_ir_datasets: _Dataset) -> None:
    root = tmp_path / "scifact"
    prepare_scifact(root)
    prepare_scifact(root, force=True)
    assert fake_ir_datasets.load_count == 2


def test_a_subset_cache_is_not_reused_for_a_full_run(
    tmp_path: Path, fake_ir_datasets: _Dataset
) -> None:
    """The bug this guards: a smoke run leaving a truncated corpus behind."""
    root = tmp_path / "scifact"
    subset = prepare_scifact(root, max_docs=3, max_queries=1)
    assert subset.subset is True

    full = prepare_scifact(root)
    assert fake_ir_datasets.load_count == 2
    assert full.subset is False
    assert full.doc_count > subset.doc_count


def test_subset_keeps_the_documents_its_queries_need(
    tmp_path: Path, fake_ir_datasets: _Dataset
) -> None:
    prepared = prepare_scifact(tmp_path / "scifact", max_docs=3, max_queries=2)
    written = {path.stem for path in prepared.docs_dir.glob("*.txt")}
    qrels = json.loads(prepared.qrels_path.read_text(encoding="utf-8"))
    for entry in qrels["queries"]:
        for rule in entry["relevant"]:
            assert rule["external_id"] in written


def test_missing_ir_datasets_explains_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ir_datasets", None)
    with pytest.raises(ImportError, match=r"custom-rag\[eval\]"):
        prepare_scifact(tmp_path / "scifact")
