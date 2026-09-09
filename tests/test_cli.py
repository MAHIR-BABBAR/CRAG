"""End-to-end CLI behaviour: index, query, context, and eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from custom_rag import __version__
from custom_rag.cli import main

FIXTURES = Path(__file__).resolve().parent / "fixtures"
QRELS = Path(__file__).resolve().parent / "eval" / "qrels.json"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A CLI environment pointed at a throwaway index."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "hybrid.md").write_text(
        "# Hybrid retrieval\n\n"
        "BM25 and dense results are fused with reciprocal rank fusion.\n\n"
        "## Why\n\nLexical search catches exact identifiers like ZenithProtocol.\n",
        encoding="utf-8",
    )
    (corpus / "notes.txt").write_text(
        "Parent chunks give the model surrounding context.\n", encoding="utf-8"
    )
    monkeypatch.setenv("CRAG_STORAGE_PATH", str(tmp_path / "crag.sqlite3"))
    monkeypatch.setenv("CRAG_COLLECTION", "cli")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-embed-v1")
    monkeypatch.delenv("CRAG_API_KEY", raising=False)
    from custom_rag.core.config import get_settings

    get_settings.cache_clear()
    yield corpus
    get_settings.cache_clear()


def test_version_flag() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_index_then_query(workspace: Path) -> None:
    runner = CliRunner()
    indexed = runner.invoke(main, ["index", str(workspace), "--provider", "mock"])
    assert indexed.exit_code == 0, indexed.output
    assert "indexed: 2" in indexed.output
    assert "failed: 0" in indexed.output

    queried = runner.invoke(
        main,
        ["query", "ZenithProtocol", "--provider", "mock", "--mode", "hybrid", "--json"],
    )
    assert queried.exit_code == 0, queried.output
    payload = json.loads(queried.output)
    assert payload["hits"]
    assert "hybrid.md" in json.dumps(payload) or payload["hits"][0]["doc_id"]


def test_context_emits_citations(workspace: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(main, ["index", str(workspace), "--provider", "mock"]).exit_code == 0

    result = runner.invoke(
        main,
        ["context", "reciprocal rank fusion", "--provider", "mock", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["citations"]
    assert payload["retrieval_quality"] in {"empty", "low", "high"}
    assert payload["context"]


def test_query_against_empty_index_is_not_an_error(workspace: Path) -> None:
    result = CliRunner().invoke(
        main, ["query", "anything", "--provider", "mock", "--mode", "hybrid", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["hits"] == []


def test_index_rejects_a_missing_path(workspace: Path) -> None:
    result = CliRunner().invoke(main, ["index", str(workspace / "missing.md")])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_index_records_files_it_cannot_parse(workspace: Path) -> None:
    (workspace / "notes.unknownext").write_text("no parser handles this", encoding="utf-8")
    (workspace / "broken.json").write_text("{not valid json", encoding="utf-8")
    result = CliRunner().invoke(main, ["index", str(workspace), "--provider", "mock"])
    assert result.exit_code == 0, result.output
    # The unknown extension is skipped by the walker; the malformed JSON is a
    # parse failure, and must be reported rather than swallowed.
    assert "failed: 1" in result.output
    assert "broken.json" in result.output


def test_eval_fixtures_suite_writes_report(tmp_path: Path, workspace: Path) -> None:
    report_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        main,
        [
            "eval",
            "--suite",
            "fixtures",
            "--fixtures",
            str(FIXTURES),
            "--qrels",
            str(QRELS),
            "--provider",
            "mock",
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report_path.is_file()
    assert report_path.with_suffix(".md").is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(payload["modes"]) == {"vector", "bm25", "hybrid"}
    assert payload["run"]["grading"]["unit"] == "document"
