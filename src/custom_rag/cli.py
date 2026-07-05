"""CRAG command-line interface."""

import click

from custom_rag import __version__


@click.group()
@click.version_option(version=__version__, prog_name="crag")
def main() -> None:
    """Custom RAG — index, retrieve, and serve."""


if __name__ == "__main__":
    main()
