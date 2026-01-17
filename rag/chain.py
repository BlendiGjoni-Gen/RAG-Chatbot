from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from rag.prompt import get_rag_prompt
from rag.retriever import get_retriever


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"--- DOCUMENT START ---\n"
            f"IDENTIFIER: {source}\n"
            f"PAGE: {page}\n"
            f"CONTENT: {doc.page_content}\n"
            f"--- DOCUMENT END ---"
        )
    return "\n\n".join(formatted)


def rag_chain(question: str, llm: BaseChatModel) -> str:
    retriever = get_retriever()

    # Retrieve relevant documents
    docs = retriever.invoke(question)
    print("RETRIEVED DOCS:", len(docs))
    for i, d in enumerate(docs[:5]):
        print(f"\n--- DOC {i} ---\n", d.page_content[:500])

    # If no docs are retrieved, refuse to answer
    if not docs or not is_relevant(question, docs):
        return "I don't know. The information is not available in the provided documents."

    context = format_docs(docs)
    prompt = get_rag_prompt()

    messages = prompt.format_messages(context=context, question=question)

    # Force stop sequences to prevent hallucinated extra questions
    response = llm.invoke(
        messages,
        stop=["DONE"]
    )

    answer = response.content.strip()

    # Extra safety: if model somehow ignored context
    if answer.lower().startswith("i don't know") or len(docs) == 0:
        return "I don't know. The information is not available in the provided documents."

    return answer

def is_relevant(question: str, docs: List[Document], min_overlap: int = 2) -> bool:
    question_terms = set(question.lower().split())

    for doc in docs:
        content_terms = set(doc.page_content.lower().split())
        if len(question_terms & content_terms) >= min_overlap:
            return True

    return False
