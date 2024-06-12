import os

from rag.utils.configurables import (
    llm_modelname_configurable,
    llm_temperature_configurable,
)

from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
TEMPERATURE = os.environ.get("TEMPERATURE", 0.5)

openai_llm = ChatOpenAI(
    model_name=MODEL_NAME, temperature=TEMPERATURE
).configurable_fields(
    model_name=llm_modelname_configurable(), temperature=llm_temperature_configurable()
)

# TODO add a configurable_alternative LLM
