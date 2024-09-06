import os
import click
from asf_hp_installer_chatbot import config
from typing import Optional


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
@click.pass_context
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


@cli.command(
    name="test_query",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
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
    "--query",
    help="Query to test.",
    type=Optional[str],
)
@click.pass_context
def test_retrieval_query(
    ctx,
    url: str,
    collection_name: str,
    query: Optional[str],
):
    """Test the retrieval of a query"""
    try:
        from rag.tests.test_retrieval import test_retrieval_query, query_at_end

        extra_kwargs = {
            ctx.args[i][2:]: ctx.args[i + 1] for i in range(0, len(ctx.args), 2)
        }

        test_retrieval_query(
            query=query if query is not None else query_at_end,
            url=url,
            collection_name=collection_name,
            **extra_kwargs,
        )
    except Exception as e:
        click.echo(f"Exception: {e}")


if __name__ == "__main__":
    cli()
