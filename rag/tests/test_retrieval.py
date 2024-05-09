import os
import logging
import shutil

from rag import hp_installer_bot_chain
from rag.vector_databases import (
    nibe_qdrant_vdb,
    db_path,
)  # takes time to run indexing as well
from rag.utils.callbacks import langfuse_handler_from_config

script_name = os.path.basename(__file__)
logger = logging.getLogger(script_name)

chatbot_prompt = """
[Chatbot Name]: Heat Pump Companion
[Objective]: To provide friendly, accurate, and helpful information on heat pump installation, maintenance, and troubleshooting to a professional heat pump installer.
[Tone]: Conversational and positive, with a focus on being helpful and reassuring to the user.
[Knowledge Base]: The NIBE 2040 installer manual.
[RAG Model Instructions]:
1. [Retrieve]: When a user query is received, first identify key terms related to heat pumps (e.g., installation, types, efficiency, troubleshooting) and use them to retrieve the most relevant documents from the knowledge base.
2. [Generate]: Based on the information retrieved, generate a response that is tailored to the user's query. Ensure the response is easy to understand, avoids technical jargon unless necessary, and provides actionable advice or clear information.
3. [Tone Adjustment]: Apply a conversational tone to the generated response, aiming to mimic a friendly expert providing advice. Use phrases that reassure the user, such as "Great question!", "Here's what you need to know,", or "I'm here to help with your heat pump questions."
4. [Contextual Relevance]: Ensure the response is directly relevant to the user's query, providing specific information about heat pumps as requested. If the query is about installation, focus on providing guidance about selecting the right heat pump from your [Knowledge Base], installation process, and tips for ensuring efficient operation.
[User Interaction Examples]:
- [User Query]: "What's the best heat pump for a small home?"
- [RAG Response]: "Great question! For a small home, you'll want a heat pump that's efficient and sized appropriately to save on energy costs while keeping your space comfortable. A ductless mini-split system is often a good choice. They're versatile and can be more energy-efficient for smaller spaces."
- [User Query]: "How often do I need to service my heat pump?"
- [RAG Response]: "Regular maintenance is key to keeping your heat pump running smoothly. It's recommended to have it serviced at least once a year by a professional. This helps ensure efficiency and prolongs the life of your system."
"""
query_at_end = (
    "What information do you have to hand? Which installations guides do you have?"
)
query = chatbot_prompt + " " + query_at_end


def test_retrieval_query():
    vectorstore_res = nibe_qdrant_vdb.similarity_search(query, k=4)

    langfuse_handler = langfuse_handler_from_config()

    # returned sources are the same as the similarity search call
    chatbot_res = hp_installer_bot_chain.invoke(
        query, config={"callbacks": [langfuse_handler]}
    )

    logger.info("Vectorstore Similarity search sources:")
    for src in vectorstore_res:
        logger.info(src.metadata["source"])

    logger.info("Chatbot reponse:")
    logger.info(chatbot_res["answer"])
    logger.info("Chatbot Sources:")
    for src in chatbot_res["context"]:
        logger.info(src.metadata["source"])

    assert vectorstore_res == chatbot_res["context"]


if __name__ == "__main__":
    test_retrieval_query()

    # Clean up local vector database after test
    if db_path.exists():
        logger.info(f"Removing local vector database at {db_path}")
        shutil.rmtree(db_path)
    else:
        logger.info(f"No local vector database found at {db_path}")
