from typing import Dict, Any

import numpy as np

from app.utils.callbacks import langfuse_handler
from app.utils.logging import logger

from rag.utils.callbacks import init_ragas_metrics

from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization

# init evaluation metrics
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
    return result
