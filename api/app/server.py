from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.runnables import RunnableConfig

from app.utils.callbacks import langfuse_handler
from app.utils.events import lifespan
from app.utils.evals import run_evaluation
from rag.chains import rag_chain_with_source

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

with rag_chain_with_source() as chatbot_chain:
    add_routes(
        app,
        chatbot_chain.with_config(RunnableConfig(callbacks=[langfuse_handler])),
        disabled_endpoints=["playground"],
    )


@app.post("/score")
def score(trace_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Score a trace with RAGAS.
    This endpoint takes a trace_id and scores the trace with RAGAS metrics: faithfulness, answer_relevancy, and context_utilization.

    The evaluation is done asynchronously using FastAPI's BackgroundTasks.

    Args:
        trace_id (str): The trace_id of the trace to be scored
        background_tasks (BackgroundTasks): FastAPI background task manager

    Returns:
        JSONResponse: Message indicating that scoring the trace with trace_id was requested
    """
    background_tasks.add_task(run_evaluation, trace_id)

    return JSONResponse(
        content=f"Requested to evaluate trace: {trace_id}",
        media_type="application/json",
    )


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
