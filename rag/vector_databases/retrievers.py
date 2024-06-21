from rag.vector_databases import init_vdb
from rag.utils.configurables import (
    retriever_searchtype_configurable,
    retriever_searchkwargs_configurable,
)

chatbot_retriever = (
    init_vdb()
    .as_retriever()
    .configurable_fields(
        search_type=retriever_searchtype_configurable(),
        search_kwargs=retriever_searchkwargs_configurable(),
    )
)
