"""This is essentially a testing script to test out different retrieval methods in the pipeline and does not involve
having to spin out the WhatsApp bot. It also quickly enables us to see the output of the retrieval methods and see if
the database has the correct information.
It will also enable us to evaluate the output in a quick way. It uses a lot of the functions from heat_pump_chatbot_post_embedding.py

This script consists of:
- Define the query to be used for the retrieval methods.
- Perform a similarity search using the Pinecone vectorstore.
- Create a retrieval QA model with sources.
- Run the retrieval QA model without sources.

To run this script, execute the following command:
'python test_vectorstore.py'
"""

import os
import openai
import pinecone
from asf_hp_installer_chatbot.pipeline.heat_pump_chatbot_post_embedding import (
    get_vector_embeddings_df,
    init_pinecone,
    create_and_initialize_index,
    get_openai_embeddings,
    upsert_data_to_index,
    get_chat_openai,
    get_retrieval_qa,
    get_pinecone_vectorstore,
)
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
import logging

logging.basicConfig(level=logging.INFO)

# Set the OpenAI API key and Pinecone environment
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("gcp-starter") or "gcp-starter"
# Model parameters, input file and chatbot prompt
index_name = "chatbot-onboarding"
model_name = "text-embedding-ada-002"
output_file = "/outputs/embedding/vector_embeddings_nibe_f2040_231844-5.pkl"
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


def perform_similarity_search(vectorstore: Pinecone, query: str, k: int = 3):
    """
    Performs a similarity search on the given vectorstore with the provided query.

    Args:
        vectorstore: The vectorstore to perform the search on.
        query: The query to use for the search.
        k (int, optional): The number of most relevant documents to return. Defaults to 3.

    Returns:
        The result of the similarity search.
    """
    return vectorstore.similarity_search(
        query, k=k  # our search query  # return k most relevant docs
    )


def create_retrieval_qa_with_sources(
    llm: ChatOpenAI, chain_type: str, vectorstore: Pinecone
) -> RetrievalQAWithSourcesChain:
    """
    Creates a RetrievalQAWithSourcesChain object from the specified chain type, llm, and vectorstore.

    Args:
        llm: The language model to use for the RetrievalQAWithSourcesChain.
        chain_type (str): The type of chain to create.
        vectorstore: The vectorstore to use as the retriever for the RetrievalQAWithSourcesChain.

    Returns:
        RetrievalQAWithSourcesChain: The created RetrievalQAWithSourcesChain object.
    """
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type=chain_type, retriever=vectorstore.as_retriever()
    )


if __name__ == "__main__":
    vector_embeddings_df = get_vector_embeddings_df(output_file)
    init_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
    index = create_and_initialize_index(index_name)
    upsert_data_to_index(index, vector_embeddings_df)
    embed = get_openai_embeddings(model_name)
    index = pinecone.Index(index_name)
    text_field = "text"
    vectorstore = get_pinecone_vectorstore(index, embed)
    sim_search_output = perform_similarity_search(vectorstore, query)
    logging.info("Similarity search output:")
    logging.info(sim_search_output)
    print(sim_search_output)
    llm = get_chat_openai()
    qa = get_retrieval_qa(llm, vectorstore)
    response = qa.run(query)
    logging.info("Chatbot response without sources")
    logging.info(response)
    qa_with_sources = create_retrieval_qa_with_sources(llm, "stuff", vectorstore)
    response_source = qa_with_sources(query)
    logging.info("Chatbot response with sources:")
    logging.info(response_source["answer"])
    logging.info("Sources:")
    logging.info(response_source["sources"])
