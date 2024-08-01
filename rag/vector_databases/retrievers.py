from langchain_core.vectorstores import VectorStoreRetriever

from rag.vector_databases import init_vdb
from rag.utils.configurables import (
    retriever_searchtype_configurable,
    retriever_searchkwargs_configurable,
)


def init_chatbot_retriever(**kwargs) -> VectorStoreRetriever:
    """
    Initializes a retriever for the chatbot Qdrant vector document store, with search_type and search_kwargs as configurable fields.

    Returns:
        VectorStoreRetriever: Qdrant vector store retriever with search_type and search_kwargs configurable fields.
    """
    return (
        init_vdb(**kwargs)
        .as_retriever()
        .configurable_fields(
            search_type=retriever_searchtype_configurable(),
            search_kwargs=retriever_searchkwargs_configurable(),
        )
    )
