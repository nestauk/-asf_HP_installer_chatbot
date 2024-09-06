import os
import click
from asf_hp_installer_chatbot import config


@click.group(name="chatbot")
@click.pass_context
def cli(ctx):
    pass


@cli.command("init_vdb")
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Initialize a local vector database. Defaults to False, using a Remote connection instead.",
)
@click.option(
    "--url",
    default=os.environ.get("QDRANT_URL", "http://qdrant-vdb:6334"),
    help="URL of the Qdrant server.",
)
@click.option(
    "--collection_name",
    default=config["index_name"],
    help="Name of the collection to use.",
)
@click.option(
    "--doc_directory",
    default=config["doc_dir"],
    help="Directory containing documents to load.",
)
@click.option(
    "--recreate", is_flag=True, default=False, help="Recreate the collection."
)  # TODO implement generalised recreation of collections
@click.option(
    "--pdf_ingest_mode",
    default="paged",
    help="PDF ingestion mode. Defaults to 'paged'.",
)
@click.option(
    "--max_workers",
    default=10,
    help="Maximum number of workers to use for document loading.",
)
@click.option(
    "--chunk_docs",
    is_flag=True,
    default=False,
    help="Chunk documents into sentences.",
)
def init_vdb_cmd(
    local: bool,
    url: str,
    collection_name: str,
    doc_directory: str,
    recreate: bool,
    pdf_ingest_mode: str = "paged",
    max_workers: int = 10,
    chunk_docs: bool = False,
):
    """Initialize a Qdrant vector database collection."""
    try:
        from rag.vector_databases.databases import init_vdb

        init_vdb(
            local=local,
            url=url,
            collection_name=collection_name,
            doc_directory=doc_directory,
            recreate=recreate,
            pdf_ingest_mode=pdf_ingest_mode,
            max_workers=max_workers,
            chunk_docs=chunk_docs,
        )
    except Exception as e:
        click.echo(f"Exception: {e}")


if __name__ == "__main__":
    cli()
