from asf_hp_installer_chatbot import Path, config, doc_dir
from rag.utils import Document
from rag.utils.pdf import CustomPDFLoader

from typing import List, Union

from concurrent.futures import ProcessPoolExecutor
from itertools import chain


def load_transformed_docs(
    doc_directory: Union[Path, str] = doc_dir,
    mode: str = "paged",
    max_workers: int = 10,
    chunk_docs: bool = False,
) -> List[Document]:
    """Load and transform heatpump manuals and guides to chunks with NLTK sentence tokenizer.

    Args:
        doc_directory (Union[Path, str], optional): Directory containing heatpump manuals and guides. Defaults to `config["doc_dir"]`.
    Returns:
        List[Document]: Heatpump Manuals and Guides as chunked Langchain Documents.
    """
    directory = Path(doc_directory)

    loaders = [
        CustomPDFLoader(file_path=file, mode=mode) for file in directory.glob("*.pdf")
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        res = [
            executor.submit(loader.load, chunk_docs=chunk_docs) for loader in loaders
        ]
        docs = list(chain(*[r.result() for r in res]))

    return docs
