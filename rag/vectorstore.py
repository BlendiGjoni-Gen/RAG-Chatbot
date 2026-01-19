from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag.embeddings import get_embedding_model

VECTORSTORE_DIR = Path("data/vectorstore")

#Create vector store and save it in folder data/vectorstore using FAISS
def create_vectorstore(chunks: List[Document]) -> FAISS:
    embedding_model = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return vectorstore

#Load the vector store
def load_vectorstore() -> FAISS:
    embedding_model = get_embedding_model()
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(f"vectorstore directory {VECTORSTORE_DIR} does not exist")

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embedding_model,
        allow_dangerous_deserialization=True,
    )