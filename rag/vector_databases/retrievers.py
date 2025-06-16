from langchain_core.vectorstores import VectorStoreRetriever, VectorStore

from contextlib import contextmanager

from typing import Generator, Any

from rag.vector_databases import init_vdb
from rag.utils.configurables import (
    retriever_searchtype_configurable,
    retriever_searchkwargs_configurable,
)


def init_configurable_retriever(vdb: VectorStore) -> VectorStoreRetriever:
    return vdb.as_retriever().configurable_fields(
        search_type=retriever_searchtype_configurable(),
        search_kwargs=retriever_searchkwargs_configurable(),
    )


@contextmanager
def init_chatbot_retriever(**kwargs) -> Generator[VectorStoreRetriever, Any, None]:
    """
    Initializes a retriever for the chatbot Qdrant vector document store, with search_type and search_kwargs as configurable fields.

    Returns:
        VectorStoreRetriever: Qdrant vector store retriever with search_type and search_kwargs configurable fields.
    """
    with init_vdb(**kwargs) as vdb:
        yield init_configurable_retriever(vdb=vdb)
