"""
PDF Processor Script

This script supports operations to extract text from PDF files, tokenize the text into chunks,
process these chunks through an embedding model, and organize the generated embeddings along with
their metadata. It assumes the presence of specific libraries (`fitz` for PDF processing, `nltk` for
text tokenization, and an openai API client for embeddings) and a directory `self.pdf_dir` where
PDF files are stored.
"""

from openai import OpenAI
import nltk
import fitz  # PyMuPDF
import os
from typing import List, Tuple


# Fetch the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class PDFProcessor:
    """
    Provides functionality for processing PDF files, including text extraction, text chunking,
    and embedding generation with metadata association.

    Attributes:
        pdf_dir (str): Directory containing PDF files to process.

    Methods:
        extract_text_from_pdfs: Extracts texts and filenames from PDFs in a specified directory.
        chunk_text: Splits text into sentence chunks and tags each with metadata.
        process_pdfs: Processes PDFs to extract text, chunk it, and tag chunks with metadata.
        create_embeddings_alpha: Generates embeddings for text chunks, tagging them with metadata.
    """

    def __init__(self, pdf_dir: str):
        """
        Initialises a new instance of the PDFProcessor class.

        Args:
            pdf_dir (str): Directory containing PDF files to process.
        """
        self.pdf_dir = pdf_dir

    def extract_text_from_pdfs(self) -> Tuple[List[str], List[str]]:
        """
        Extracts and returns text from PDF files in a specified directory along with their filenames.

        Iterates over PDF files in `self.pdf_dir`, extracting text from each and compiling a list of texts and
        corresponding filenames (as metadata tags).

        Returns:
            tuple[list[str], list[str]]: A pair of lists containing the extracted texts and their respective filenames.

        Note:
            Depends on the `fitz` (PyMuPDF) module for reading PDFs. Only processes files ending in '.pdf'.
        """
        pdf_texts = []
        pdf_metadata_tags = []
        dir_contents = os.listdir(self.pdf_dir)
        list_of_pdfs_paths = [
            os.path.join(self.pdf_dir, f) for f in dir_contents if f.endswith(".pdf")
        ]
        for pdf_file in list_of_pdfs_paths:
            with fitz.open(pdf_file) as pdf_document:
                pdf_text = ""
                for page in pdf_document:
                    pdf_text += page.get_text()
                pdf_texts.append(pdf_text)
                pdf_file_name = pdf_file.replace(self.pdf_dir, "")
                pdf_metadata_tags.append(pdf_file_name)
        return pdf_texts, pdf_metadata_tags

    def chunk_text(self, text: str, metadata_tag: str) -> Tuple[List[str], List[str]]:
        """
        Tokenizes text into sentences and associates each with a given metadata tag.

        Args:
            text (str): Text to be chunked.
            metadata_tag (str): Tag to associate with each text chunk.

        Returns:
            A tuple containing two lists:
            - First, a list of sentence chunks from the text.
            - Second, a list of metadata tags corresponding to each chunk.

        Requires the `nltk.sent_tokenize` method from NLTK for sentence tokenization.
        """
        text_chunks = nltk.sent_tokenize(text)
        chunk_metadata_tags = [metadata_tag] * len(text_chunks)
        return text_chunks, chunk_metadata_tags

    def process_pdfs(self) -> Tuple[List[str], List[str]]:
        """
        Processes PDF files by extracting, chunking texts, and associating chunks with their source PDFs' metadata tags.

        First, extracts text from each PDF file in a specified directory. Then, splits the text from each PDF into
        sentence chunks and associates these chunks with the PDF's filename. Aggregates all chunks and their corresponding
        metadata tags into lists and returns them.

        Returns:
            tuple[list[str], list[str]]: A pair of lists containing all text chunks and their respective metadata tags.
        """
        texts, metadata_tags = self.extract_text_from_pdfs()
        chunked_texts = []
        chunked_metadata_tags = []
        for text, metadata_tag in zip(texts, metadata_tags):
            chunks, chunk_tags = self.chunk_text(text, metadata_tag)
            chunked_texts.extend(chunks)
            chunked_metadata_tags.extend(chunk_tags)
        return chunked_texts, chunked_metadata_tags

    def create_embeddings_alpha(
        self, texts: List[str], chunk_tags: List[str], model_id: str, pdf_web_dict: dict
    ) -> Tuple[List, List[dict], List[str]]:
        """
        Generates embeddings for text chunks, associates each with metadata, and assigns unique identifiers.

        Processes text chunks to generate embeddings using the specified model. Constructs metadata for each chunk,
        including its sequence number within its source document, the source PDF's tag, and additional source information
        from a provided mapping. Generates unique identifiers for each embedding based on document and chunk sequence.

        Args:
            texts (list[str]): The list of text chunks to embed.
            chunk_tags (list[str]): Corresponding tags identifying the source PDF for each text chunk.
            model_id (str): Identifier for the embedding model to use.
            pdf_web_dict (dict): Mapping of PDF tags to additional source information.

        Returns:
            tuple[list, list[dict], list[str]]: A tuple containing:
                - A list of embeddings.
                - A list of dictionaries, each representing metadata for a corresponding embedding.
                - A list of unique identifiers for each embedding, based on document and chunk sequence.
        """
        embeddings = []
        metadata_list = []  # This will contain our metadata with chunk and source
        document_counter = 1  # Starting with the first document
        chunk_counter = 0  # Initialize chunk counter
        prev_tag = None  # Keep track of the previous tag
        for text, tag in zip(texts, chunk_tags):
            response = client.embeddings.create(
                input=text, model=model_id
            )  # Use your chosen model ID)
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            # Increment document counter if the tag changes (indicating a new document)
            if tag != prev_tag:
                document_counter += 1 if prev_tag is not None else 0
                chunk_counter = 0  # Reset chunk counter for a new document
                prev_tag = tag  # Update the previous tag
            # Create the metadata with chunk number and the PDF name from the tag
            metadata = {
                "chunk": chunk_counter,
                "pdf source": tag,
                "source": pdf_web_dict[tag],
                "text": text,
            }
            metadata_list.append(metadata)
            # Increment the chunk counter
            chunk_counter += 1
        # Create the ids based on the document and chunk counters
        ids = [f"{document_counter}-{i}" for i in range(len(embeddings))]
        return embeddings, metadata_list, ids
