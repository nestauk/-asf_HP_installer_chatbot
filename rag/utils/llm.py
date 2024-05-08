from rag.utils.configurables import (
    llm_modelname_configurable,
    llm_temperature_configurable,
)

from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0.5
).configurable_fields(
    model_name=llm_modelname_configurable(), temperature=llm_temperature_configurable()
)

# TODO add a configurable_alternative LLM
