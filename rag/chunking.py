from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


#Functions to chunk the docs and clean them for later use
def chunk_docs(documents: List[Document], chunk_size: int = 1000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', '')}_{chunk.metadata.get('start_index', 0)}"
    return chunks

def clean_chunks(docs: List[Document]) -> List[Document]:
    cleaned_docs = []
    for doc in docs:
        text = "\n".join(
            line for line in doc.page_content.splitlines()
            if not line.strip().startswith("Question:") and not line.strip().startswith("[/ASS]")
        )
        new_doc = Document(
            page_content=text,
            metadata=doc.metadata
        )
        cleaned_docs.append(new_doc)
    return cleaned_docs