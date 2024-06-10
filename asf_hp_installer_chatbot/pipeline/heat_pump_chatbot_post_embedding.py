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
import yaml
from typing import Any, Callable
from asf_hp_installer_chatbot import PROJECT_DIR
from asf_hp_installer_chatbot import config

# Set the OpenAI API key and Pinecone environment
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("gcp-starter") or "gcp-starter"


# Read in CSV file with vector embeddings and metadata
def get_vector_embeddings_df(
    output_file: str = config["most_recent_embedding"],
) -> pd.DataFrame:
    """
    Loads and returns a DataFrame of vector embeddings from a pickle file.

    Args:
        output_file (str): The path to the pickle file relative to the project directory.
        Defaults to the most recent embedding.

    Returns:
        pd.DataFrame: A DataFrame containing vector embeddings.
    """
    return pd.read_pickle(os.path.join(PROJECT_DIR, output_file))


def init_pinecone():
    """
    Initialises Pinecone using the provided API key and environment.
    This function is used to initialise the Pinecone service, which is a vector database service that allows efficient
    storage and retrieval of high-dimensional vectors.
    Note:
        Depends on the `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` variables.
    """
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


def create_and_initialize_index(
    index_name: str = config["index_name"],
) -> pinecone.GRPCIndex:
    """
    Creates and initialises a Pinecone index with the specified name.

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


def get_openai_embeddings(model_name: str = config["model_id"]) -> OpenAIEmbeddings:
    """
    Returns an OpenAIEmbeddings object for the specified model.

    This function is used to create an OpenAIEmbeddings object, which is a wrapper around the OpenAI API that
    provides functionality for generating embeddings from text data. Embeddings are high-dimensional vector
    representations of text that capture semantic meaning.
    Args:
        model_name (str): The name of the model to use for generating embeddings. This should be the name of a pre-trained model provided by the OpenAI

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


def get_chat_openai(
    model: str = config["gpt_model"], temp: float = config["temp"]
) -> ChatOpenAI:
    """
    Creates and returns a ChatOpenAI object configured for interacting with OpenAI's Chat API.

    Args:
        model (str, optional): The model name determines which version of GPT (Generative Pre-trained Transformer)
                               will be used for generating responses.
        temp (float, optional): The temperature parameter controls the randomness of the output, with lower values
                                producing more deterministic and predictable text, and higher values resulting in
                                more varied and creative responses. Suggested range is between 0 and 1.


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
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ID", "Question", "Answer", "Timestamp"])  # header
        writer.writerow([sender_hash, incoming_msg, answer, timestamp])


def create_bot(
    vectorstore: Any,
    chatbot_prompt: str = config["chatbot_prompt"],
    output_Q_and_A: str = config["Q_and_A_file"],
) -> Callable:
    """
    Factory function that creates a Flask route function (bot) for handling incoming messages,
    generating responses, and appending the data to a CSV file.

    Args:
        vectorstore (Any): The storage for vector representations of the knowledge base.
        chatbot_prompt (str): The chatbot's prompt that will be used as a part of the input to the model.
        output_Q_and_A (str): The path to the CSV file where the Q&A data will be stored.

    Returns:
        Callable: A Flask route function (bot) that handles incoming messages, generates responses,
                  and appends the data to a CSV file. The bot function uses the request object from Flask
                  to get the incoming message and sender's number from the HTTP request, and returns the
                  chatbot's response in TwiML format, which can be returned as a response to a Twilio webhook.
    """

    def bot():
        incoming_msg = request.values.get("Body", "").lower()
        sender_number = request.values.get("From", "")
        # Generate a SHA256 hash of the sender number
        sender_hash = get_sender_hash(sender_number)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create a new instance of the MessagingResponse class. This will be used to generate the TwiML response.
        resp = MessagingResponse()
        # Create a new <Message> element that will be added to the MessagingResponse. This will contain the chatbot's response.
        msg = resp.message()
        # Call the get_chat_openai function to get a language model from OpenAI. The gpt_model and temperature parameters are used to configure the model.
        llm = get_chat_openai()
        # Call the get_retrieval_qa function to create a QA system. The llm and vectorstore parameters are used to configure the system.
        qa = get_retrieval_qa(llm, vectorstore)
        # Generate a response to the incoming message. The chatbot_prompt and incoming_msg are concatenated and passed to the run method of the QA system.
        response = qa.run(chatbot_prompt + " " + incoming_msg)
        # Assign the generated response to the answer variable. This seems redundant in this context, unless answer is used elsewhere in the code.
        answer = response
        # Set the body of the <Message> element to the generated response. This is the message that will be sent back to the user.
        msg.body(response)
        # Append the data to a CSV file
        append_to_csv(
            f"{PROJECT_DIR}/{output_Q_and_A}",
            sender_hash,
            incoming_msg,
            answer,
            timestamp,
        )
        return str(resp)

    return bot


if __name__ == "__main__":
    vector_embeddings_df = get_vector_embeddings_df()
    init_pinecone()
    index = create_and_initialize_index()
    upsert_data_to_index(index, vector_embeddings_df)
    embed = get_openai_embeddings()
    # switch back to normal index for langchain
    index_name = config["index_name"]
    langchain_index = pinecone.Index(index_name)
    vectorstore = get_pinecone_vectorstore(langchain_index, embed)
    app = Flask(__name__)
    bot_with_args = create_bot(vectorstore)
    app.route("/bot", methods=["POST"])(bot_with_args)
    app.run(port=4000)
