"""
The create_vector_embeddings.py script processes PDF files to extract text, creates text embeddings, and stores these embeddings along with metadata in a DataFrame and outputs a pickle file.

It performs the following steps:
1. Initializes a PDFProcessor instance with a directory of PDF files.
2. Processes the PDFs to extract text and chunk them, associating chunks with metadata tags.
3. Generates embeddings for each text chunk using the OpenAI API.
4. Collects embeddings, their metadata, and unique identifiers into a DataFrame.
5. Saves the DataFrame as a pickle file for later use.

To run this script, execute the following command:
'python create_vector_embeddings.py'

"""

from pdf_processor import PDFProcessor
import os
import openai
import pandas as pd
from asf_hp_installer_chatbot import PROJECT_DIR


if __name__ == "__main__":
    # Fetch the API key from the environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Define the directory containing the installation pdfs
    pdf_dir = (
        f"{PROJECT_DIR}/inputs/data/air2water_HP_installation_guides/database_pdf/"
    )

    # Initialise PDFProcessor and process PDFs
    pdf_processor = PDFProcessor(pdf_dir)
    chunked_texts, chunked_metadata_tags = pdf_processor.process_pdfs()

    # Define a mapping from installation pdfs to their online sources
    pdf_to_web_dict = {
        "Nibe_F2040_231844-5.pdf": "https://www.nibe.eu/assets/documents/16900/231844-5.pdf"
    }

    # Set the model ID for embedding generation
    model_id = "text-embedding-ada-002"

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

    # Generate the output file name based on installation pdfs
    pdf_names = "_".join([name.replace(".pdf", "") for name in pdf_to_web_dict.keys()])
    # Save the DataFrame as a pickle file
    output_file = f"{PROJECT_DIR}/outputs/embedding/vector_embeddings_{pdf_names}.pkl"
    vector_embeddings_df.to_pickle(output_file)
