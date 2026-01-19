from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)

#Functions to load the docs (pdf, markdown, docs) and create metadata
def load_pdfs(directory: str) -> List[Document]:
    documents = []
    pdf_files = Path(directory).glob("*.pdf")

    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        for page_num, page in enumerate(pages):
            page.metadata["source"] = pdf_path.name
            page.metadata["page"] = page_num + 1
            documents.append(page)

    return documents


def load_markdown(directory: str) -> List[Document]:
    documents = []
    md_files = Path(directory).glob("*.md")

    for md_path in md_files:
        loader = UnstructuredMarkdownLoader(str(md_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = md_path.name
            documents.append(doc)

    return documents


def load_documents(directory: str) -> List[Document]:
    if not Path(directory).exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents = []
    documents.extend(load_pdfs(directory))
    documents.extend(load_markdown(directory))

    return documents