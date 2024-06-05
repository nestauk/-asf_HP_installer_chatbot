import os, logging

from fastapi import FastAPI

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    """Asyncronous context manager for the lifespan of the FastAPI application. Manages the startup and shutdown of the application.

    Args:
        app (FastAPI): The FastAPI application.
    """
    # https://fastapi.tiangolo.com/advanced/events/#lifespan-function
    # Lines before yield statement execute when the application starts.
    # Add any pre-processing logic here.

    # TODO: investigate why logging not working for server deployment. assumption - uvicorn overrides logging
    logger = logging.getLogger("events.lifespan")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    if os.environ.get("NGROK_AUTHTOKEN") is not None:
        import ngrok

        listener = await ngrok.forward(8000, authtoken_from_env=True)
        logger.info(f"Ingress established at {listener.url()}")

    if os.environ.get("QDRANT_URL") is not None:
        logger.info("QDRANT_URL is set. Using hosted qdrant server.")
    else:
        logger.info("QDRANT_URL not set. Using local qdrant server.")

    yield
    # Lines after yield statement execute when the application stops.

    await listener.close()

    from rag.vector_databases import db_path

    if db_path.exists():
        import shutil

        shutil.rmtree(db_path)
