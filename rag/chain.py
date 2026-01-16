from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag.prompt import get_rag_prompt
from rag.retriever import get_retriever

def format_docs(docs: List[Document]) -> str:

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        formatted.append(
            f"{doc.page_content}\n"
            f"(Source: {source}, page {page})"
        )

    return "\n".join(formatted)

def rag_chain(question: str, llm: BaseChatModel) -> str:

    retriever = get_retriever()

    docs = retriever.invoke(question)

    if not docs:
        return "I don't know. The information is not available in the provided documents."

    context = format_docs(docs)

    prompt = get_rag_prompt()

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    answer = response.content.strip()

    if "I don't know" in answer and "Source" not in answer:
        return "I don't know. The information is not available in the provided documents."

    return answer