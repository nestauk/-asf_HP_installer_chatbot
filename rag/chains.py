from typing import Generator, Any
from contextlib import contextmanager
from functools import partialmethod

from rag.vector_databases.retrievers import init_chatbot_retriever
from rag.prompt_templates import chatbot_template, chatbot_with_history_template
from rag.utils.llm import openai_llm
from rag.utils.text_processing import format_source_docs

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableSerializable,
)

from rag.utils.history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory

rag_chain = chatbot_template | openai_llm | StrOutputParser()

rag_with_history_chain = chatbot_with_history_template | openai_llm | StrOutputParser()


@contextmanager
def rag_with_source_base(**kwargs) -> Generator[RunnableParallel, Any, None]:
    """Initializes a chatbot chain with a context retriever. Kwargs are passed to init_vdb and the retriever.

    Returns:
        RunnableParallel: Chatbot chain with context retriever.
    """
    with init_chatbot_retriever(**kwargs) as retriever:
        yield (
            RunnableParallel(
                {
                    "context": retriever | format_source_docs,
                    "query": RunnablePassthrough(),
                }
            )
        )


@contextmanager
def rag_chain_with_source(**kwargs) -> Generator[RunnableSerializable, Any, None]:
    """Initializes a chatbot chain with a source retriever. Kwargs are passed to init_vdb and the retriever. This chain makes it possible to retrieve the source documents from the retriever by referencing the "context" key in the output.

    Returns:
        RunnableSerializable: Chatbot chain with source retriever.
    """
    with rag_with_source_base(**kwargs) as rag_with_source:
        yield rag_with_source.assign(answer=rag_chain)


def chatbot_with_history_chain(**kwargs) -> RunnableSerializable:
    return rag_with_source_base(**kwargs).assign(
        answer=RunnableWithMessageHistory(
            rag_with_history_chain,
            get_session_history=get_session_history,
            input_messages_key="query",
            history_messages_key="history",
        )
    )
