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
def init_vdb_cmd(
    local: bool,
    url: str,
    collection_name: str,
    doc_directory: str,
    recreate: bool,
):
    """Initialize a Qdrant vector database collection."""
    try:
        from rag.vector_databases.databases import init_vdb

        init_vdb(local, url, collection_name, doc_directory, recreate)
    except Exception as e:
        click.echo(f"Exception: {e}")


if __name__ == "__main__":
    cli()
