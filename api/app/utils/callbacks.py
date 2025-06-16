from rag.utils.callbacks import langfuse_handler_from_config

langfuse_handler = langfuse_handler_from_config(
    trace_name="hp_installer_chatbot_chain",
    user_id="testing",  # TODO: change to UUID
    session_id="local",  # TODO: change to UUID + Date
    version="0.1.0",  # TODO Customise these/ move to .env
    release="0.1.0",  # TODO Customise these/ move to .env
    tags=["dev", "api", "v1"],  # TODO Customise these/ move to .env
)
