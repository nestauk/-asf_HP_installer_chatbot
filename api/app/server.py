from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.runnables import RunnableConfig

from app.utils.events import lifespan
from rag.chains import rag_chain_with_source
from rag.utils.callbacks import langfuse_handler_from_config, init_ragas_metrics

from app.client import chat

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization

import sys
import logging


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

langfuse_handler = langfuse_handler_from_config(
    trace_name="hp_installer_chatbot_chain",
    user_id="testing",
    session_id="local",
    version="0.1.0",
    release="0.1.0",
    tags=["dev", "api", "v1"],
)  # TODO move to env vars

app.add_route("/hook", chat, methods=["POST"])

chatbot_chain = rag_chain_with_source().with_config(
    RunnableConfig(callbacks=[langfuse_handler])
)

add_routes(
    app,
    chatbot_chain,
    # path="/chat", # renames first span from RunnableSequence to "/chat"
    # disabled_endpoints=["playground"] # temporarily disabled due to security risk
)

## Evaluation
init_ragas_metrics([faithfulness, answer_relevancy, context_utilization])


@app.post("/score")
def score(trace_id: str) -> JSONResponse:
    """
    Score a trace with RAGAS.
    This function is defined synchronous as the RAGAS evaluate() function does not support uvloop at this time.
    This endpoint takes a trace_id (str) and responds with the scores for the trace (JSONResponse).

    The scores response has the following SUCCESS schema,

    {
        "trace_id": str,
        "scores": {
            "context_utilization": float,
            "faithfulness": float,
            "answer_relevancy": float
        }
    }

    and the following ERROR schema,

    {
        "error": str
    }

    Args:
        trace_id (str): The trace ID to score

    Returns:
        JSONResponse: The scores for the trace or error.
    """
    # Get the trace
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
                retriever_context = [doc["page_content"] for doc in retrieved_chunks]
            if obs.name == "RunnableAssign<answer>":
                output: dict = obs.output
                assert output is not None, "RunnableAssign<answer> output is None"
                context: str = output.get("context", "")
                query: str = output.get("query", "")
                answer: str = output.get("answer", "")

        ## Sense check data can be evaluated
        # LLM responses are chatty and will not be empty unless there was an error
        assert answer != "", "RunnableAssign<answer> output.answer is empty"
        # Value from VectorStoreRetriever output should be the same as the context from RunnableAssign<answer>
        assert context == "\n\n".join(retriever_context), "Contexts do not match"
    except AssertionError as e:
        response = f"Data could not be evaluated: {e}"
        logger.error(response)
        return JSONResponse(
            content=jsonable_encoder({"error": response}),
            status_code=400,
            media_type="application/json",
        )

    # Construct evaluation dataset
    evaluation_batch = {
        "question": [query],
        "contexts": [retriever_context if retriever_context else [context]],
        "answer": [answer],
    }
    ds = Dataset.from_dict(evaluation_batch)

    # Score with RAGAS - cannot be run with uvloop
    res = (
        evaluate(ds, [faithfulness, answer_relevancy, context_utilization])
        .to_pandas()
        .to_dict()
    )

    response = {
        "trace_id": trace_id,
        "scores": {},
    }

    # Send score to Langfuse
    metric_names = ["context_utilization", "faithfulness", "answer_relevancy"]
    for metric_name, metric_value in res.items():
        if metric_name in metric_names:
            logger.info(f"Sending {metric_name} score to Langfuse: {metric_value[0]}")
            langfuse_handler.langfuse.score(
                trace_id=trace_id,
                name=metric_name,
                value=metric_value[0],
                comment=f"Ground-truthless RAGAS LLM-based score",
            )
            response["scores"][metric_name] = metric_value[0]

    return JSONResponse(
        content=jsonable_encoder(response),
        media_type="application/json",
    )


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
