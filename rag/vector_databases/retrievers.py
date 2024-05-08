from rag.vector_databases.databases import nibe_qdrant_vdb
from rag.utils.configurables import (
    retriever_searchtype_configurable,
    retriever_searchkwargs_configurable,
)

nibe_manual_retriever = nibe_qdrant_vdb.as_retriever().configurable_fields(
    search_type=retriever_searchtype_configurable(),
    search_kwargs=retriever_searchkwargs_configurable(),
)
