from fastapi import FastAPI, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.runnables import RunnableConfig

from app.utils.callbacks import langfuse_handler
from app.utils.events import lifespan
from rag.chains import rag_chain_with_source

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization

import os, sys
import logging
from typing import Dict, Any

import numpy as np


def get_logger():
    logger = logging.getLogger("HPInstallerChatbotAPI")
    logger.setLevel(logging.INFO)
    logging.StreamHandler(sys.stdout)
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logger


logger = get_logger()

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
    )

## Evaluation
context_utilization = ContextUtilization()
metrics = [faithfulness, answer_relevancy, context_utilization]
init_ragas_metrics(metrics)


async def get_trace_data(trace_id: str) -> Dict[str, Any]:
    observations = [
        obs
        for obs in langfuse_handler.langfuse.fetch_trace(trace_id).data.observations
        if obs.name in ["VectorStoreRetriever", "RunnableAssign<answer>"]
    ]

    try:
        # Get the query, context, and answer
        for obs in observations:
            if obs.name == "VectorStoreRetriever":
                retrieved_chunks = (res for res in obs.output)
                retriever_context = [str(doc) for doc in retrieved_chunks]
            if obs.name == "RunnableAssign<answer>":
                output: dict = obs.output
                assert output is not None, "RunnableAssign<answer> output is None"
                context: str = str(output.get("context", ""))
                query: str = output.get("query", "")
                answer: str = output.get("answer", "")

        ## Sense check data can be evaluated
        # LLM responses are chatty and will not be empty unless there was an error
        assert answer != "", "RunnableAssign<answer> output.answer is empty"
        # Value from VectorStoreRetriever output should be the same as the context from RunnableAssign<answer>
        # assert context == "\n\n".join(retriever_context), "Contexts do not match"
    except AssertionError as e:
        response = f"Data could not be evaluated: {e}"
        logger.error(response)

    evaluation_batch = {
        "question": query,
        "contexts": retriever_context if retriever_context else context,
        "answer": answer,
    }

    return evaluation_batch


async def evaluate_response(
    query: str, contexts: list, answer: str
) -> Dict[str, float]:
    scores = {}
    for m in metrics:
        scores[m.name] = await m.ascore(
            row={"question": query, "contexts": contexts, "answer": answer}
        )
    return scores


async def send_scores(trace_id: str, result: Dict[str, float]) -> None:
    for metric_name, metric_value in result.items():
        # Ensure NaN values are converted Python floats
        score_value = (
            np.nan_to_num(metric_value) if np.isnan(metric_value) else metric_value
        )

        logger.info(f"Sending {metric_name} score to Langfuse: {score_value}")
        langfuse_handler.langfuse.score(
            trace_id=trace_id,
            name=metric_name,
            value=score_value,
            type="NUMERIC",
            comment=f"Ground-truthless RAGAS LLM-based score",
        )


async def run_evaluation(trace_id: str) -> Dict[str, float]:
    trace_data = await get_trace_data(trace_id)
    result = await evaluate_response(
        query=trace_data["question"],
        contexts=trace_data["contexts"],
        answer=trace_data["answer"],
    )

    await send_scores(trace_id, result)


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
