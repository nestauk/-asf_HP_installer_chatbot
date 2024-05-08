from rag.utils import Document
from rag.utils.pdf import CustomPDFLoader

from typing import List


def load_transformed_docs() -> List[Document]:
    """Load and transform NIBE 2040 installer manual to chunks with NLTK sentence tokenizer.

    Returns:
        List[Document]: NIBE 2040 installer manual as chunked Langchain Documents.
    """
    loader = CustomPDFLoader("https://www.nibe.eu/assets/documents/16900/231844-5.pdf")
    docs = loader.load(chunk_docs=True)
    return docs
