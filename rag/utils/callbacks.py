import json
import logging
from uuid import UUID
from typing import Optional, Any, Dict, List, Sequence

import asyncio
from functools import lru_cache # may need this for caching metric objects

from datasets import Dataset

from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# RAGAS imports
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

# Project imports
from rag.utils.llm import ChatOpenAI, openai_llm
from rag.utils.embeddings import OpenAIEmbeddings, openai_embeddings

def langfuse_handler_from_config(
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    return CallbackHandler(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        version=version,
        release=release,
        tags=tags,
    )

# util function to init Ragas Metrics
def init_ragas_metrics(
    metrics: List[Any],
    llm: ChatOpenAI = openai_llm,
    embeddings: OpenAIEmbeddings = openai_embeddings
):
    eval_callback = langfuse_handler_from_config(
        trace_name="EvaluationRun",
        user_id="LLMEvaluator",
        session_id="API",
        version="0.1.0",
        release="0.1.0",
        tags=["evaluation", "ragas", "dev", "api", "v1"],
    )
    config = RunnableConfig(callbacks=[eval_callback])

    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = LangchainLLMWrapper(llm.with_config(config))
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = LangchainEmbeddingsWrapper(embeddings)
        run_config = RunConfig()
        metric.init(run_config)

class EvaluationCallbackHandler(CallbackHandler):
    langfuse_client = Langfuse()
    logger = logging.getLogger("EvaluationCallback")
    logger.setLevel(logging.INFO)
    eval_data = {}

    def __init__(self, metrics: List[Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        ## Setup RAGAS evaluation
        self.metrics = metrics
        init_ragas_metrics(
            metrics=self.metrics,
            llm=openai_llm,
            embedding=openai_embeddings,
        )

    async def score_with_ragas(self, trace_id: str, query, chunks, answer):
        scores = {}
        for m in self.metrics:
            self.logger.info(f"Calculating {m.name}")
            scores[m.name] = await m.ascore(
                row={"question": query, "contexts": chunks, "answer": answer}
            )

            # Send score to Langfuse
            self.langfuse_client.score(
                trace_id=trace_id, name=m.name, value=scores[m.name]
            )
        return scores

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        super().on_retriever_end(documents, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.eval_data["context"] = [doc.page_content for doc in documents]

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.eval_data["query"] = str(outputs.get("query"))
        self.eval_data["answer"] = str(outputs.get("answer"))
        self.eval_data["run_id"] = str(run_id)

        # Evaluate - this takes a while
        asyncio.run(
            self.score_with_ragas(
                trace_id=self.eval_data["run_id"],
                query=self.eval_data["query"],
                chunks=self.eval_data["context"],
                answer=self.eval_data["answer"],
            )
        )

def eval_handler_from_config(
    metrics: List[Any],
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    return EvaluationCallbackHandler(
        metrics=metrics,
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        version=version,
        release=release,
        tags=tags,
    )
