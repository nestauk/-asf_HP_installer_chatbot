from rag.vector_databases import nibe_manual_retriever
from rag.prompt_templates import chatbot_template, chatbot_with_history_template
from rag.utils.llm import openai_llm
from rag.utils.text_processing import format_source_docs

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from rag.utils.history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory

rag_chain = chatbot_template | openai_llm | StrOutputParser()

rag_with_source_base = RunnableParallel(
    {
        "context": nibe_manual_retriever | format_source_docs,
        "query": RunnablePassthrough(),
    }
)

rag_chain_with_source = rag_with_source_base.assign(
    answer=rag_chain
)  # Chain assignments aren't dynamically configurable yet

rag_with_history_chain = chatbot_with_history_template | openai_llm | StrOutputParser()

chatbot_with_history_chain = rag_with_source_base.assign(
    answer=RunnableWithMessageHistory(
        rag_with_history_chain,
        get_session_history=get_session_history,
        input_messages_key="query",
        history_messages_key="history",
    )
)
