"""This module provides functions to evaluate responses in a RAGAS system using Langfuse. It integrates with the Langfuse framework to fetch trace data, evaluate responses using RAGAS metrics, and send evaluation scores back to Langfuse.
It includes functions to fetch trace data, evaluate responses, and send scores to Langfuse.

It defines the `get_trace_data`, `evaluate_response`, `send_scores`, and `run_evaluation` functions.

It uses the RAGAS metrics for evaluation, including faithfulness, answer relevancy, and context utilization.

This module is typically used in conjunction with the RAGAS metrics and Langfuse handler to provide a comprehensive evaluation framework.
"""

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
    """Fetches the trace data from Langfuse for a given trace ID and extracts the relevant information for evaluation.

    This function retrieves the trace observations, specifically looking for the "VectorStoreRetriever" and "RunnableAssign<answer>" observations.
    It then extracts the query, context, and answer from these observations to create an evaluation batch.
    This function is used to prepare the data for evaluation by the RAGAS metrics.
    It ensures that the data is in the correct format and checks for any inconsistencies or errors in the retrieved data.
    If any issues are found, it logs an error message and raises an assertion error.
    This function is essential for ensuring that the evaluation process has the necessary data to compute the RAGAS metrics accurately.

    Example usage:
        trace_id = "example_trace_id"
        evaluation_data = await get_trace_data(trace_id)
        print(evaluation_data)
    This function is typically called as part of the evaluation process in the RAGAS framework.
    It is used to fetch the trace data from Langfuse, extract the relevant information, and prepare it for evaluation by the RAGAS metrics.
    This function is asynchronous and returns a dictionary containing the extracted trace data for evaluation.
    It is designed to be used in conjunction with the RAGAS metrics to evaluate the performance of a trace in a RAGAS system.
    It is important to ensure that the trace data is correctly formatted and contains the necessary information for evaluation.

    Args:
        trace_id (str): The ID of the trace to fetch data for.

    Returns:
        Dict[str, Any]: The extracted trace data for evaluation.
    """
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
    """Evaluate the response using RAGAS metrics.
    This function takes a query, contexts, and an answer, and evaluates the response using the RAGAS metrics.
    It computes the scores for each metric defined in the `metrics` list and returns a dictionary containing the scores.
    This function is used to evaluate the performance of a trace in a RAGAS system.
    It is typically called after fetching the trace data from Langfuse and extracting the relevant information for evaluation.
    It is designed to be used in conjunction with the `get_trace_data` function to prepare the data for evaluation.
    This function is asynchronous and returns a dictionary containing the scores for each metric.
    It is important to ensure that the query, contexts, and answer are correctly formatted and contain the necessary information for evaluation.
    Example usage:
        query = "What is the capital of France?"
        contexts = ["Paris is the capital of France.", "France is a country in Europe."]
        answer = "The capital of France is Paris."
        scores = await evaluate_response(query, contexts, answer)
        print(scores)

    Args:
        query (str): The query string to evaluate.
        contexts (list): A list of context strings to consider during evaluation.
        answer (str): The answer string to evaluate.

    Returns:
        Dict[str, float]: A dictionary containing the scores for each metric.
    """
    scores = {}
    for m in metrics:
        scores[m.name] = await m.ascore(
            row={"question": query, "contexts": contexts, "answer": answer}
        )
    return scores


async def send_scores(trace_id: str, result: Dict[str, float]) -> None:
    """Send the evaluation scores to Langfuse for the given trace ID.
    This function takes a trace ID and a dictionary of evaluation scores, and sends each score to Langfuse.
    It uses the `langfuse_handler` to send the scores, ensuring that NaN values are converted to Python floats before sending.
    This function is used to log the evaluation scores for a trace in a RAGAS system.
    It is typically called after evaluating the response using the `evaluate_response` function.
    It is designed to be used in conjunction with the `run_evaluation` function to send the scores after evaluation.
    This function is asynchronous and does not return any value.
    Example usage:
        trace_id = "example_trace_id"
        result = {"faithfulness": 0.85, "answer_relevancy": 0.90, "context_utilization": 0.75}
        await send_scores(trace_id, result)
    This function is essential for logging the evaluation scores in a RAGAS system.
    It ensures that the scores are sent to Langfuse for tracking and analysis.
    It is important to ensure that the trace ID and result dictionary are correctly formatted and contain the necessary information for logging.
    This function is typically called as part of the evaluation process in the RAGAS framework.
    It is used to send the evaluation scores for a trace to Langfuse, allowing for tracking and analysis of the trace's performance in a RAGAS system.

    Args:
        trace_id (str): The ID of the trace to log.
        result (Dict[str, float]): A dictionary containing the evaluation scores for the trace.
    """
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
    """Run the evaluation for a given trace ID.
    This function fetches the trace data using the `get_trace_data` function, evaluates the response using the `evaluate_response` function,
    and sends the scores to Langfuse using the `send_scores` function.
    It is designed to be used as a high-level function that orchestrates the evaluation process in a RAGAS system.
    It is typically called after a trace has been created and the relevant data has been logged in Langfuse.
    This function is asynchronous and returns a dictionary containing the evaluation scores for the trace.
    Example usage:
        trace_id = "example_trace_id"
        evaluation_result = await run_evaluation(trace_id)
        print(evaluation_result)
    This function is essential for running the evaluation process in a RAGAS system.
    It ensures that the trace data is fetched, evaluated, and the scores are sent to Langfuse for tracking and analysis.
    It is important to ensure that the trace ID is correctly formatted and corresponds to a valid trace in Langfuse.

    Args:
        trace_id (str): The ID of the trace to evaluate.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation scores for the trace.
    """
    trace_data = await get_trace_data(trace_id)
    result = await evaluate_response(
        query=trace_data["question"],
        contexts=trace_data["contexts"],
        answer=trace_data["answer"],
    )

    await send_scores(trace_id, result)
    return result
