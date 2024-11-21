import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional, Any

from asf_hp_installer_chatbot import config, doc_dir, db_path

from rag.utils.embeddings import openai_embeddings
from rag.utils.data_loaders import load_transformed_docs

from langchain_community.vectorstores.qdrant import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.http.exceptions import UnexpectedResponse
from grpc._channel import _InactiveRpcError

logger = logging.getLogger("qdrant_vdb")


@contextmanager
def local_vdb(
    collection_name: str = config["index_name"],
    doc_directory: str = doc_dir,
    mode: str = "paged",
    max_workers: int = 10,
    chunk_docs: bool = False,
) -> Generator[Qdrant, Any, None]:
    index_params = {
        "embedding": openai_embeddings,
        "path": db_path,
        "collection_name": collection_name,
    }
    try:
        client = QdrantClient(path=db_path)

        if client.collection_exists(collection_name):
            logger.info(f"Collection: {collection_name} already exists.")
            client.close()

            vdb: Qdrant = Qdrant.from_existing_collection(**index_params)
            logger.info(f"Using collection: {collection_name}")

            yield vdb
        else:
            logger.info(f"Collection: {collection_name} does not exist.")
            client.close()

            logger.info("Loading documents...")
            docs = load_transformed_docs(
                doc_directory=doc_directory,
                mode=mode,
                max_workers=max_workers,
                chunk_docs=chunk_docs,
            )
            logger.info(f"Number of documents loaded: {len(docs)}")

            logger.info(f"Creating new local collection: {collection_name}")
            vdb: Qdrant = Qdrant.from_documents(
                documents=docs,
                **index_params,
            )
            logger.info(f"Created new local collection: {collection_name}.")

            yield vdb
    except Exception as e:
        logger.error(f"Exception: {e}")


def remote_vdb(
    url: str = os.environ.get("QDRANT_URL", "http://qdrant-vdb:6334"),
    collection_name: str = config["index_name"],
    doc_directory: str = doc_dir.as_posix(),
    recreate=False,
    mode: str = "paged",
    max_workers: int = 10,
    chunk_docs: bool = False,
    prefer_grpc: bool = False,
    api_key: Optional[str] = None,
    https: bool = False,
) -> Qdrant:
    client_params = {
        "url": url,
        "collection_name": collection_name,
    }

    if recreate:
        client_params["force_recreate"] = True
        logger.info(f"Recreating collection: {collection_name}")

        logger.info("Loading documents...")
        docs = load_transformed_docs(
            doc_directory=doc_directory,
            mode=mode,
            max_workers=max_workers,
            chunk_docs=chunk_docs,
        )
        logger.info(f"Number of documents loaded: {len(docs)}")

        vdb = Qdrant.from_documents(
            documents=docs,
            embedding=openai_embeddings,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https,
            **client_params,
        )

        return vdb

    try:
        vdb = Qdrant.from_existing_collection(
            embedding=openai_embeddings,
            path=None,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https,
            **client_params,
        )
        logger.info("Connected to vector database.")

        try:
            client = QdrantRemote(
                url=url, prefer_grpc=prefer_grpc, api_key=api_key, https=https
            )
            logger.info(f"Checking if collection exists: {collection_name}")
            client.get_collection(collection_name)
        except (UnexpectedResponse, _InactiveRpcError) as e:
            logger.error(f"Exception: {e}")
            logger.info(
                f"An Exception occured or, Collection does not exist: {collection_name}"
            )
            logger.info("Creating a new collection...")
            vdb = Qdrant.from_documents(
                documents=load_transformed_docs(
                    doc_directory=doc_directory,
                    mode=mode,
                    max_workers=max_workers,
                    chunk_docs=chunk_docs,
                ),
                embedding=openai_embeddings,
                prefer_grpc=prefer_grpc,
                api_key=api_key,
                https=https,
                **client_params,
            )

        logger.info(f"Collection exists: {collection_name}")
        logger.info(f"Using collection: {collection_name}")

    except Exception as e:
        logger.error(f"Exception: {e}")

    return vdb


@contextmanager
def init_vdb(
    local: bool = False,
    url: str = os.environ.get("QDRANT_URL", "http://qdrant-vdb"),
    collection_name: str = config["index_name"],
    doc_directory: str = doc_dir.as_posix(),
    recreate: bool = False,
    pdf_ingest_mode: str = "paged",
    max_workers: int = 10,
    chunk_docs: bool = False,
    prefer_grpc: bool = False,
    api_key: Optional[str] = None,
    https: bool = False,
) -> Generator[Qdrant, Any, None]:

    if ":6334" in url:
        prefer_grpc = True

    scheme = "https" if https else "http"
    if "http" not in url:
        url = f"{scheme}://{url}"
    else:
        url = url.replace("http://", f"{scheme}://").replace("https://", f"{scheme}://")

    try:
        if local:
            if recreate:
                raise ValueError(
                    "Cannot recreate a local collection. Remove the `--recreate` flag."
                )
            with local_vdb(
                collection_name,
                doc_directory=doc_directory,
                mode=pdf_ingest_mode,
                max_workers=max_workers,
                chunk_docs=chunk_docs,
            ) as vdb:
                yield vdb
        else:
            vdb = remote_vdb(
                url=url,
                collection_name=collection_name,
                recreate=recreate,
                doc_directory=doc_directory,
                mode=pdf_ingest_mode,
                max_workers=max_workers,
                chunk_docs=chunk_docs,
                prefer_grpc=prefer_grpc,
                api_key=api_key,
                https=https,
            )
            yield vdb
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
    finally:
        vdb.client.close()
        logger.info("Closed connection to vector database.")
