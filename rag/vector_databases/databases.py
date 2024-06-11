import os

from asf_hp_installer_chatbot import PROJECT_DIR

from rag.utils.embeddings import openai_embeddings
from rag.utils.data_loaders import load_transformed_docs

from langchain_community.vectorstores.qdrant import Qdrant

db_path = PROJECT_DIR / "outputs/data/vector_db/qdrant/local_test"
# TODO add deployment config for a vectordb instance, or optionally change vector database back to Pinecone

if (URL := os.environ.get("QDRANT_URL")) is None:
    vdb_path_params = {"path": db_path, "collection_name": "nibe_manual"}
else:
    # TODO add URL format validation
    vdb_path_params = {
        "url": URL,
        "prefer_grpc": True,
        "collection_name": "nibe_manual",
    }

nibe_qdrant_vdb = Qdrant.from_documents(
    documents=load_transformed_docs(), embedding=openai_embeddings, **vdb_path_params
)
