"""
The create_vector_embeddings.py script processes PDF files to extract text, creates text embeddings, and stores these embeddings along with metadata in a DataFrame and outputs a pickle file.

It performs the following steps:
1. Initialises a PDFProcessor instance with a directory of PDF files.
2. Processes the PDFs to extract text and chunk them, associating chunks with metadata tags.
3. Generates embeddings for each text chunk using the OpenAI API.
4. Collects embeddings, their metadata, and unique identifiers into a DataFrame.
5. Saves the DataFrame as a pickle file for later use.

To run this script, execute the following command:
'python create_vector_embeddings.py'

"""

import logging
from pdf_processor import PDFProcessor
import os
import yaml
import openai
import pandas as pd
from asf_hp_installer_chatbot import PROJECT_DIR
from asf_hp_installer_chatbot import config
from datetime import datetime
import hashlib
import json
from typing import Tuple, Dict, List, Union


def update_metadata_file(metadata: Dict, metadata_file: str):
    """
    Update the metadata file with new metadata.

    This function loads the existing metadata from the file, updates it with the new metadata,
    and then writes the updated metadata back to the file.

    Args:
        metadata (dict): The new metadata to add to the file. This should be a dictionary where
            the keys are timestamps and the values are lists of PDF names.
        metadata_file (str): The path to the metadata file.

    Note:
        If the metadata file does not exist or does not contain valid JSON, an empty dictionary
        will be used as the existing data.
    """
    # Try to load existing data
    try:
        with open(metadata_file, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning(
            """Metadata file not found or invalid JSON. Creating new metadata file."""
        )
        existing_data = {}

    # Add the new data
    existing_data.update(metadata)

    # Write the data back to the file
    with open(metadata_file, "w") as f:
        json.dump(existing_data, f)


def create_metadata(
    pdf_to_web_dict: Dict,
) -> Tuple[Dict[str, Union[str, List[str]]], str, str]:
    """
    Creates a metadata dictionary with a timestamp, list of PDF names, and a hash of the PDF names.

    Args:
        pdf_to_web_dict (Dict): A dictionary mapping PDF names to web URLs.

    Returns:
        Tuple: A tuple containing the following elements:
            - metadata (Dict): A dictionary with the following keys:
                - "timestamp": The current date and time as a string in the format "YYYYMMDD_HHMMSS".
                - "pdf_names": A list of the keys of pdf_to_web_dict.
                - "hash": A SHA256 hash of the sorted keys of pdf_to_web_dict, truncated to the first 20 characters.
            - timestamp (str): The current date and time as a string in the format "YYYYMMDD_HHMMSS".
            - hash_hex_short (str): A SHA256 hash of the sorted keys of pdf_to_web_dict, truncated to the first 20 characters.
    """
    # Get the current date and time
    now = datetime.now()
    # Format as a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # Create a dictionary mapping the timestamp to the list of PDF names
    keys = pdf_to_web_dict.keys()
    keys_str = str(sorted(keys))
    # Create a SHA256 hash of the keys string
    hash_object = hashlib.sha256(keys_str.encode())
    hash_hex = hash_object.hexdigest()
    # Truncate the hash to the first 20 characters
    hash_hex_short = hash_hex[:20]
    metadata = {
        "timestamp": timestamp,
        "pdf_names": list(pdf_to_web_dict.keys()),
        "hash": hash_hex_short,
    }
    return metadata, timestamp, hash_hex_short


if __name__ == "__main__":
    # Fetch the API key from the environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Define the directory containing the installation pdfs
    pdf_dir = os.path.join(PROJECT_DIR, config["pdf_dir"])
    print(pdf_dir)
    # Initialise PDFProcessor and process PDFs
    pdf_processor = PDFProcessor(pdf_dir)
    chunked_texts, chunked_metadata_tags = pdf_processor.process_pdfs()

    # Define a mapping from installation pdfs to their online sources
    pdf_to_web_dict = config["pdf_to_web_dict"]

    # Set the model ID for embedding generation
    model_id = config["model_id"]

    # Generate embeddings and metadata
    embeddings, metadata_list, ids = pdf_processor.create_embeddings_alpha(
        chunked_texts, chunked_metadata_tags, model_id, pdf_to_web_dict
    )

    # Create a DataFrame to store the embeddings, metadata, and unique identifiers
    vector_embeddings_df = pd.DataFrame(
        {
            "id": ids,  # Use the ids generated from the create_embeddings function
            "values": embeddings,
            "metadata": metadata_list,  # Renamed from 'blobs' to 'metadata'
        }
    )
    # Get the current date and time
    metadata, timestamp, hash_hex_short = create_metadata(pdf_to_web_dict)
    # Write the metadata to a JSON file
    metadata_file = f"{PROJECT_DIR}/outputs/embedding/pdf_embedding_metadata.json"
    # Update the metadata file with the new data
    update_metadata_file(metadata, metadata_file)
    # Output the vector embeddings to a pickle file
    base_output_file = (
        f"outputs/embedding/vector_embeddings_{hash_hex_short}_{timestamp}.pkl"
    )
    output_file = os.path.join(PROJECT_DIR, base_output_file)
    vector_embeddings_df.to_pickle(output_file)
    config["most_recent_embedding"] = base_output_file
    with open(f"{PROJECT_DIR}/asf_hp_installer_chatbot/config/base.yaml", "w") as file:
        yaml.dump(config, file)
