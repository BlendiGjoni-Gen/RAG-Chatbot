from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder
from typing import List

from rag.vectorstore import load_vectorstore

def get_retriever(k: int = 6) -> VectorStoreRetriever:

    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": 0.5  # Ignore everything below 50% match
        },
    )


def rerank_documents(query: str, documents: List[Document], top_n: int = 3):
    """Rerank retrieved documents using cross-encoder."""
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    # Sort by score
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_n]]