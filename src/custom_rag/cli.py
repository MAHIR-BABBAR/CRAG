"""CRAG command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from custom_rag import __version__


@click.group()
@click.version_option(version=__version__, prog_name="crag")
def main() -> None:
    """Custom RAG — index, retrieve, and serve."""


@main.command("parse")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit ParsedDocument as JSON.")
def parse(path: Path, as_json: bool) -> None:
    """Parse a document and print block summary or JSON."""
    from custom_rag.ingestion import parse_file

    document = parse_file(path)
    if as_json:
        click.echo(json.dumps(document.model_dump(mode="json"), indent=2))
        return

    click.echo(f"doc_type: {document.metadata.doc_type}")
    click.echo(f"blocks: {len(document.blocks)}")
    click.echo(f"content_hash: {document.metadata.content_hash}")
    for block in document.blocks[:10]:
        preview = block.text[:80].replace("\n", " ")
        click.echo(f"  [{block.block_type}] {block.block_id}: {preview}")
    if len(document.blocks) > 10:
        click.echo(f"  ... and {len(document.blocks) - 10} more blocks")


if __name__ == "__main__":
    main()
