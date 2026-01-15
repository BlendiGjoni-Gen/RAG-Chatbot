from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from rag.vectorstore import load_vectorstore

def get_retriever(k: int = 4) -> VectorStoreRetriever:

    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )