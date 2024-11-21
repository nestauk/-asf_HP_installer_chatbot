import os
import click
from asf_hp_installer_chatbot import config, doc_dir
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
    help="URL of the Qdrant server. Defaults to 'http://qdrant-vdb:6334'.",
)
@click.option(
    "--collection_name",
    default=config["index_name"],
    help="Name of the collection to use. Defaults to the index_name in config.yaml.",
)
@click.option(
    "--doc_directory",
    default=doc_dir.as_posix(),
    help="Directory containing documents to load. Defaults to the doc_dir project.",
)
@click.option(
    "--recreate", is_flag=True, default=False, help="Recreate the collection."
)  # TODO implement generalised recreation of collections
@click.option(
    "--pdf_ingest_mode",
    default="paged",
    help="PDF ingestion mode. Defaults to 'paged'. Options: 'paged', 'full'. Defaults to 'paged'.",
)
@click.option(
    "--max_workers",
    default=10,
    help="Maximum number of workers to use for document loading. Defaults to 10.",
)
@click.option(
    "--chunk_docs",
    is_flag=True,
    default=False,
    help="Chunk documents into sentences. Defaults to False.",
)
@click.option(
    "--prefer_grpc",
    is_flag=True,
    default=False,
    help="Prefer gRPC for Qdrant server interactions. Defaults to False.",
)
@click.option(
    "--api_key",
    help="API key for Qdrant server. Defaults to None.",
)
@click.option(
    "--https",
    is_flag=True,
    default=False,
    help="Use HTTPS for Qdrant server interactions. Defaults to False.",
)
@click.pass_context
def init_vdb_cmd(
    ctx,
    local: bool,
    url: str,
    collection_name: str,
    doc_directory: str,
    recreate: bool,
    pdf_ingest_mode: str = "paged",
    max_workers: int = 10,
    chunk_docs: bool = False,
    prefer_grpc: bool = False,
    api_key: Optional[str] = None,
    https: bool = False,
):
    """Initialize a Qdrant vector database collection."""
    try:
        from rag.vector_databases.databases import init_vdb

        with init_vdb(
            local=local,
            url=url,
            collection_name=collection_name,
            doc_directory=doc_directory,
            recreate=recreate,
            pdf_ingest_mode=pdf_ingest_mode,
            max_workers=max_workers,
            chunk_docs=chunk_docs,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https,
        ) as vdb:
            assert vdb.client.collection_exists(collection_name)
            click.echo(f"Initialized collection: {collection_name}")
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
    type=str,
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Use a local vector database for testing.",
)
@click.pass_context
def test_query(
    ctx,
    url: str,
    collection_name: str,
    query: str,
    local: bool = False,
):
    """Test the retrieval of a query"""
    try:
        from rag.tests.test_retrieval import test_retrieval_query

        extra_kwargs = {
            ctx.args[i][2:]: ctx.args[i + 1] for i in range(0, len(ctx.args), 2)
        }

        test_retrieval_query(
            query=query,
            url=url,
            collection_name=collection_name,
            local=local,
            **extra_kwargs,
        )
    except Exception as e:
        click.echo(f"Exception: {e}")


if __name__ == "__main__":
    cli()
