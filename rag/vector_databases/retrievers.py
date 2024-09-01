from rag.vector_databases import init_vdb
from rag.utils.configurables import (
    retriever_searchtype_configurable,
    retriever_searchkwargs_configurable,
)

# added in doc_dir_test for testing purposes, think we can remove this and put in flag for testing
from asf_hp_installer_chatbot import doc_dir_test

chatbot_retriever = (
    init_vdb(local=True, doc_directory=doc_dir_test)
    # init_vdb(local=True)
    .as_retriever().configurable_fields(
        search_type=retriever_searchtype_configurable(),
        search_kwargs=retriever_searchkwargs_configurable(),
    )
)
