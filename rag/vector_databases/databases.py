import os
import logging
import shutil

from asf_hp_installer_chatbot import config, doc_dir

from rag.utils.embeddings import openai_embeddings
from rag.utils.data_loaders import load_transformed_docs

from langchain_community.vectorstores.qdrant import Qdrant

logger = logging.getLogger("qdrant_vdb")


def local_vdb(
    collection_name: str = config["index_name"], doc_directory: str = doc_dir
) -> Qdrant:

    try:
        from asf_hp_installer_chatbot import db_path

        if not db_path.exists():
            db_path.mkdir(parents=True)
            logger.info("Created a new local vector database directory.")
        else:
            shutil.rmtree(db_path)
            logger.info("Deleted existing local vector database directory.")

        vdb = Qdrant.from_documents(
            documents=load_transformed_docs(doc_directory=doc_directory),
            embedding=openai_embeddings,
            path=db_path,
            collection_name=collection_name,
        )
        logger.info("Created a new local vector database collection.")
        return vdb

    except Exception as e:
        logger.error(f"Exception: {e}")


def remote_vdb(
    url: str = os.environ.get("QDRANT_URL", "http://qdrant-vdb:6334"),
    collection_name: str = config["index_name"],
    doc_directory: str = doc_dir,
    recreate=False,
) -> Qdrant:
    client_params = {
        "url": url,
        "prefer_grpc": True,
        "collection_name": collection_name,
    }

    try:
        vdb = Qdrant.from_existing_collection(
            embedding=openai_embeddings, path=None, **client_params
        )
        logger.info("Connected to vector database.")

        if recreate:
            vdb.client.delete_collection(collection_name)
            raise ValueError("Requested to recreate collection", collection_name)

        if not vdb.client.collection_exists(collection_name):
            raise ValueError("Collection does not exist: ", collection_name)

        logger.info(f"Using collection: {collection_name}")

    except Exception as e:
        logger.error(f"Exception: {e}")
        logger.info("Creating a new collection...")
        vdb = Qdrant.from_documents(
            documents=load_transformed_docs(doc_directory=doc_directory),
            embedding=openai_embeddings,
            **client_params,
        )

    return vdb


def init_vdb(
    local: bool = False,
    url: str = os.environ.get("QDRANT_URL", "http://qdrant-vdb:6334"),
    collection_name: str = config["index_name"],
    doc_directory: str = doc_dir,
    recreate: bool = False,
) -> Qdrant:
    if local:
        if recreate:
            raise ValueError(
                "Cannot recreate a local collection. Remove the `--recreate` flag."
            )
        return local_vdb(collection_name, doc_directory=doc_directory)
    else:
        return remote_vdb(
            url=url,
            collection_name=collection_name,
            recreate=recreate,
            doc_directory=doc_directory,
        )
