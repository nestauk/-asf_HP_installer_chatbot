from langchain_core.runnables import ConfigurableField


def retriever_searchtype_configurable() -> ConfigurableField:
    """Return a configurable field for retriever search type.

    Returns:
        ConfigurableField: Configurable field instance for retriever search type.
    """
    return ConfigurableField(
        id="search_type", name="Search Type", description="Search type for retriever"
    )


def retriever_searchkwargs_configurable() -> ConfigurableField:
    """Return a configurable field for retriever search kwargs.

    Returns:
        ConfigurableField: Configurable field instance for retriever search kwargs.
    """
    return ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="Search arguments for retriever",
    )


def llm_modelname_configurable() -> ConfigurableField:
    """Return a configurable field for LLM model name.

    Returns:
        ConfigurableField: Configurable field instance for LLM model name.
    """
    return ConfigurableField(
        id="model_name", name="Model Name", description="Model Name for LLM"
    )


def llm_temperature_configurable() -> ConfigurableField:
    """Return a configurable field for LLM temperature.

    Returns:
        ConfigurableField: Configurable field instance for LLM temperature.
    """
    return ConfigurableField(
        id="temperature", name="Temperature", description="Temperature for LLM"
    )
