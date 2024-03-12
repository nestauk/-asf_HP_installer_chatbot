"""
We know how powerful retrieval augmentation and conversational agents (chatbots) can be. Now with the power of Pinecone and Langchain we can combine them.
Chatbots based with an LLM often struggle with data freshness, knowledge about specific domains, or accessing internal documentation. By coupling agents with retrieval augmentation tools we no longer have these problems.
On the other side, using "naive" retrieval augmentation without the use of an LLM means we will retrieve contexts with every query. Again, this isn't always ideal as not every query requires access to external knowledge.

Merging these methods gives us the best of both worlds. In this script we create a chatbot which has a specialised knowledge base of heat pump installation guides and then interface using Flask and Twilio to create a chatbot which can be accessed via WhatsApp.

The chatbot uses the following components:
1. OpenAI's GPT-3.5-turbo model for conversational responses.
2. Pinecone for efficient storage and retrieval of high-dimensional vectors.
3. Langchain for retrieval augmentation and conversational agents.

To run this script you will need to have the following environment variables set:
1. OPENAI_API_KEY: Your OpenAI API key.
2. PINECONE_API_KEY: Your Pinecone API key.
3. PINECONE_ENVIRONMENT: The name of the Pinecone environment you want to use.

To test out the WhatsApp chatbot, you can use a tool like ngrok to expose your local server to the internet:
1. Install ngrok from https://ngrok.com/download.
2. Run the following command in your terminal:
    ngrok http 4000
3. You will see a forwarding URL in the terminal. Use this URL to configure your Twilio WhatsApp sandbox.
4. Send a message to your Twilio WhatsApp sandbox number to interact with the chatbot.
5. Executing this script will start a Flask server that listens for incoming messages and responds with the chatbot's answer.

You can execute this script by running the following command in your terminal:
    python heat_pump_chatbot_post_embedding.py
"""

import os
import openai
import pandas as pd
import pinecone
import time
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import csv
import hashlib
from asf_hp_installer_chatbot import PROJECT_DIR

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


# Read in CSV file with vector embeddings and metadata
def get_vector_embeddings_df(
    output_file: str = "/outputs/embedding/vector_embeddings_nibe_f2040_231844-5.pkl",
) -> pd.DataFrame:
    """
    Loads and returns a DataFrame of vector embeddings from a pickle file.

    Args:
        output_file (str, optional): The path to the pickle file relative to the project directory.
        Defaults to '/outputs/embedding/vector_embeddings_nibe_f2040_231844-5.pkl'.

    Returns:
        pd.DataFrame: A DataFrame containing vector embeddings.
    """
    return pd.read_pickle(f"{PROJECT_DIR}{output_file}")


def init_pinecone(
    pinecone_api_key: str = PINECONE_API_KEY,
    pinecone_environment: str = PINECONE_ENVIRONMENT,
):
    """
    Initialises Pinecone using the provided API key and environment.
    This function is used to initialises the Pinecone service, which is a vector database service that allows efficient
    storage and retrieval of high-dimensional vectors.
    Note:
        Depends on the `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` variables.
    """
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)


def create_and_initialize_index(index_name: str) -> pinecone.GRPCIndex:
    """
    Creates and initializes a Pinecone index with the specified name.

    The Pinecone index is a data structure provided by the Pinecone service that allows efficient storage and retrieval
    of vectors in high-dimensional space. It is used in this function to store the vector embeddings for later retrieval.

    Args:
        index_name (str): The name of the index to create.

    Returns:
        pinecone.GRPCIndex: The created Pinecone index.
    """
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=1536,
            metadata_config={"indexed": ["chunk", "source"]},
        )
        time.sleep(1)
    return pinecone.GRPCIndex(index_name)


def upsert_data_to_index(index: pinecone.GRPCIndex, vector_embeddings_df: pd.DataFrame):
    """
    Upserts data from a DataFrame to a Pinecone index. The term "upsert" is a combination of "update" and "insert". If the data already exists in the
    index, it is updated. If it does not exist, it is inserted.

    Args:
        index (pinecone.GRPCIndex): The Pinecone index to upsert data to.
        vector_embeddings_df (pd.DataFrame): The DataFrame containing data to upsert.
    """
    index.upsert_from_dataframe(vector_embeddings_df, batch_size=100)


def get_openai_embeddings(model_name: str) -> OpenAIEmbeddings:
    """
    Returns an OpenAIEmbeddings object for the specified model.

    This function is used to create an OpenAIEmbeddings object, which is a wrapper around the OpenAI API that
    provides functionality for generating embeddings from text data. Embeddings are high-dimensional vector
    representations of text that capture semantic meaning.

    Args:
        model_name (str): The name of the model to use for generating embeddings. This should be the name of a
        pre-trained model provided by the OpenAI API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object that can be used to generate embeddings from text data. The
        object is configured to use the specified model for generating embeddings.
    """
    return OpenAIEmbeddings(model=model_name, openai_api_key=openai.api_key)


def get_pinecone_vectorstore(
    index: pinecone.GRPCIndex, embed: OpenAIEmbeddings
) -> Pinecone:
    """
    Returns a Pinecone vector store for the specified index and embeddings.

    This function is used to create a vector store, which is a data structure that allows efficient storage and
    retrieval of high-dimensional vectors, as well as other capabilities such as querying and scaling. The vector store is created using a Pinecone index and OpenAI embeddings.
    The Pinecone index is a data structure provided by the Pinecone service that allows efficient storage and
    retrieval of vectors in high-dimensional space. The OpenAI embeddings object can be used to convert text data into
    high-dimensional vectors that can be stored in the Pinecone index.

    Args:
        index (pinecone.GRPCIndex): The Pinecone index to use.
        embed (OpenAIEmbeddings): The OpenAI embeddings to use.

    Returns:
        Pinecone: A Pinecone vector store for the specified index and embeddings.
    """
    return Pinecone(index, embed.embed_query, "text")


def get_chat_openai(model: str = "gpt-3.5-turbo", temp: float = 0.5) -> ChatOpenAI:
    """
    Creates and returns a ChatOpenAI object configured for interacting with OpenAI's Chat API.

    This function configures a ChatOpenAI instance with a specific API key, model name, and temperature setting.
    The API key is required for authentication with OpenAI's services. The model name determines which version
    of GPT (Generative Pre-trained Transformer) will be used for generating responses.
    The temperature parameter controls the randomness of the output, with lower values producing more deterministic
    and predictable text, and higher values resulting in more varied and creative responses.


    Returns:
        ChatOpenAI: A ChatOpenAI object with the specified API key, model name, and temperature.
    """
    return ChatOpenAI(openai_api_key=openai.api_key, model_name=model, temperature=temp)


def get_retrieval_qa(llm: ChatOpenAI, vectorstore: Pinecone) -> RetrievalQA:
    """
    Returns a RetrievalQA object for the specified language model and vector store.
    When "Retrieval QA" uses the chain_type "stuff" in LangChain, it indicates a process where the system searches through external
    documents to find relevant information. Once this information (the "stuff") is identified, it is “stuffed” (fits within context window)
    into the LLM.
    Args:
        llm (ChatOpenAI): The language model to use.
        vectorstore (Pinecone): The vector store to use.

    Returns:
        RetrievalQA: A RetrievalQA object for the specified language model and vector store.
    """
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )


def get_sender_hash(sender_number: str) -> str:
    """
    Generates and returns a SHA256 hash of the sender's number. This is done so as to anonymise the sender's number.

    Args:
        sender_number (str): The sender's number.

    Returns:
        str: A SHA256 hash of the sender's number.
    """
    sender_number = sender_number.replace("whatsapp:", "")
    return hashlib.sha256(sender_number.encode()).hexdigest()


def get_timestamp() -> str:
    """
    Returns the current timestamp in the format 'YYYY-MM-DD HH:MM:SS'. Useful to have a timestamp to understand ordering of messages.

    Returns:
        str: The current timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_to_csv(
    filename: str, sender_hash: str, incoming_msg: str, answer: str, timestamp: str
):
    """
    Appends a row of data to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        sender_hash (str): The SHA256 hash of the sender's number.
        incoming_msg (str): The incoming message.
        answer (str): The chatbot's response.
        timestamp (str): The timestamp of the message.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ID", "Question", "Answer", "Timestamp"])  # header
        writer.writerow([sender_hash, incoming_msg, answer, timestamp])


def bot():
    """
    Handles incoming messages, generates responses, and appends the data to a CSV file.

    Returns:
        str: The chatbot's response.
    """
    incoming_msg = request.values.get("Body", "").lower()
    sender_number = request.values.get("From", "")
    # Generate a SHA256 hash of the sender number
    sender_hash = get_sender_hash(sender_number)
    sender_hash = hashlib.sha256(sender_number.encode()).hexdigest()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resp = MessagingResponse()
    msg = resp.message()
    llm = get_chat_openai()
    qa = get_retrieval_qa(llm, vectorstore)
    response = qa.run(chatbot_prompt + " " + incoming_msg)
    answer = response
    msg.body(response)
    # Append the data to a CSV file
    append_to_csv(
        f"{PROJECT_DIR}/outputs/data/incoming_messages.csv",
        sender_hash,
        incoming_msg,
        answer,
        timestamp,
    )
    return str(resp)


if __name__ == "__main__":
    vector_embeddings_df = get_vector_embeddings_df()
    init_pinecone()
    index = create_and_initialize_index(index_name)
    upsert_data_to_index(index, vector_embeddings_df)
    embed = get_openai_embeddings(model_name)
    index = pinecone.Index(index_name)
    vectorstore = get_pinecone_vectorstore(index, embed)
    app = Flask(__name__)
    app.route("/bot", methods=["POST"])(bot)
    app.run(port=4000)
