from fastapi import FastAPI

from rag.vector_databases.databases import db_path

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

    yield
    # Lines after yield statement execute when the application stops.

    if db_path.exists():
        import shutil

        shutil.rmtree(db_path)
