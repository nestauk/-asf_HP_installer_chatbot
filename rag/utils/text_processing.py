from rag.utils import Document

from langchain_text_splitters.nltk import NLTKTextSplitter

from typing import List


def nltk_splitter(
    separator: str = "\n\n",
    language: str = "english",
    chunk_size=500,
    chunk_overlap=50,
    **kwargs
) -> NLTKTextSplitter:
    """Return a NLTK sentence text splitter instance for Langchain Documents with preset config.

    Args:
        separator (str, optional): Separater to merge small sentence chunks into medium sized chunks. Defaults to "\\n\\n".
        language (str, optional): NLTK sentence tokenizer model language. Defaults to "english".
        chunk_size (int, optional): Defaults to 500.
        chunk_overlap (int, optional): Defaults to 50.

    Returns:
        NLTKTextSplitter: NLTK sentence text splitter instance with preset config.
    """
    return NLTKTextSplitter(
        separator=separator,
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )


def format_source_docs(docs: List[Document]) -> str:
    """Format a source document separated by double new lines into a single string.

    Args:
        docs (List[Document]): List of Langchain Documents.

    Returns:
        str: Formatted source document string.
    """
    return "\n\n".join(doc.page_content for doc in docs)
