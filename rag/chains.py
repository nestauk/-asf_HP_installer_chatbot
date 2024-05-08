from rag.vector_databases import nibe_manual_retriever
from rag.prompt_templates import chatbot_template
from rag.utils.llm import openai_llm
from rag.utils.text_processing import format_source_docs

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_source_docs(x["context"])))
    | chatbot_template
    | openai_llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": nibe_manual_retriever, "query": RunnablePassthrough()}
).assign(
    answer=rag_chain
)  # Chain assignments aren't configurable yet
