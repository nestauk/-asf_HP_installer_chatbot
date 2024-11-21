from asf_hp_installer_chatbot import config
from rag.utils import Document

from langchain_community.document_loaders.pdf import (
    BasePDFLoader,
    UnstructuredPDFLoader,
)

from pathlib import Path
from typing import Tuple, Union, Any, List


class CustomPDFLoader(BasePDFLoader):
    """Load single `PDF` in paged mode. Supports loading online `PDF`."""

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "paged",
        **unstructured_kwargs: Any,
    ):
        """Initialize the PDF loader."""
        super().__init__(file_path=file_path)
        self.mode = mode
        self.unstructured_kwargs = unstructured_kwargs

    def _get_source(self) -> str:
        """Return web source metadata for the PDF source.

        Returns:
            str: Web source URL metadata for the PDF source.
        """
        if self.web_path:
            path = self.web_path
        else:
            path = config["doc_source_mapping"][Path(self.file_path).name]

        return path

    def _convert_coords(self, coords: Tuple[Tuple[float]]) -> List[tuple]:
        """Convert tuple of coordinates tuple pairs to list of tuple pairs.

        Args:
            coords (Tuple[Tuple[float]]): Tuple of coordinates tuple pairs.

        Returns:
            List[tuple]: List of coordinates tuple pairs.
        """
        return [list(coord) for coord in coords]

    def _chunk_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """Load and transform documents to chunks with NLTK sentence tokenizer.

        Args:
            documents (List[Document]): List of Langchain Documents.

        Returns:
            List[Document]: Chunked Langchain Documents.
        """
        from rag.utils.text_processing import nltk_splitter

        splitter = nltk_splitter(**kwargs)
        docs = documents or self.load()
        return splitter.transform_documents(docs)

    def load(self, chunk_docs: bool = False) -> List[Document]:
        """Load documents, replaces source with web_path, or maps local filenames to it's source, if available.

        Args:
            chunk_docs (bool, optional): Option to chunk loaded documents with NLTK sentence tokenizer. Defaults to False.

        Returns:
            List[Document]: List of Langchain Documents.
        """
        source = self._get_source()
        loader = UnstructuredPDFLoader(
            file_path=str(self.file_path),
            mode=self.mode,
            unstructured_kwargs=self.unstructured_kwargs,
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = source
            doc.metadata["coordinates"]["points"] = self._convert_coords(
                doc.metadata["coordinates"]["points"]
            )

        if chunk_docs:
            return self._chunk_documents(documents=docs)

        return docs


def pdf_loader(
    file_path: Union[str, Path, List[str], List[Path]],
    mode: str = "paged",
    **unstructured_kwargs: Any,
) -> CustomPDFLoader:
    """Return a lazy PDF document loader. Supports loading online `PDF`. Load documents with `loader.load()`

    Args:
        file_path (Union[str, Path, List[str], List[Path]]): Path to the PDF file(s).
        mode (str, optional): PDF document loading mode. Paged mode adds chunk source page number metadata to each document. Defaults to "paged".

    Raises:
        NotImplementedError: Loading multiple PDFs is not yet supported.

    Returns:
        CustomPDFLoader: Lazy PDF document loader.
    """
    # TODO implement loading multiple PDFs in CustomPDFLoader
    if isinstance(file_path, list):
        raise NotImplementedError("Loading multiple PDFs is not yet supported.")

    else:
        return CustomPDFLoader(file_path=file_path, mode=mode, **unstructured_kwargs)
