from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.runnables import RunnableConfig

from app.utils.events import lifespan
from rag import hp_installer_bot_chain
from rag.utils.callbacks import langfuse_handler_from_config

from app.client import chat


app = FastAPI(lifespan=lifespan)

# CORS
origins = ["http://localhost:8000", "http://localhost"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],  # TODO: change to ["POST", "GET"] when ready
    allow_headers=["*"],
)

langfuse_handler = langfuse_handler_from_config(
    trace_name="hp_installer_chatbot_chain",
    user_id="testing",
    session_id="local",
    version="0.1.0",
    release="0.1.0",
    tags=["dev", "api", "v1"],
)  # TODO move to env vars

app.add_route("/hook", chat, methods=["POST"])

chatbot_chain = hp_installer_bot_chain.with_config(
    RunnableConfig(callbacks=[langfuse_handler])
)

add_routes(
    app,
    chatbot_chain,
    # path="/chat", # renames first span from RunnableSequence to "/chat"
    # disabled_endpoints=["playground"] # temporarily disabled due to security risk
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
