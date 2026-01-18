import streamlit as st
from rag.chain import rag_chain
from rag.llm import get_llm
from rag.retriever import get_retriever
from rag.vectorstore import load_vectorstore
from pathlib import Path
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("📚 RAG Chatbot")
st.write(
    """
Ask questions based on the uploaded knowledge base.
The bot will only answer using the documents in the vectorstore.
"""
)

# -----------------------------
# Streamlit caching to speed up loading
# -----------------------------
@st.cache_resource
def load_retriever_and_vectorstore(k: int = 4):
    BASE_DIR = Path(__file__).resolve().parent.parent
    vectorstore_path = BASE_DIR / "data" / "vectorstore"

    if not vectorstore_path.exists():
        st.error("Vectorstore not found. Please run the ingestion pipeline first.")
        return None

    retriever = get_retriever(k=k)
    return retriever

@st.cache_resource
def load_llm_instance():
    llm = get_llm()
    return llm

llm = load_llm_instance()

# -----------------------------
# User input
# -----------------------------
question = st.text_input("Enter your question:")

if st.button("Ask") and question.strip():
    with st.spinner("Fetching answer..."):
        # Run RAG chain
        answer = rag_chain(question, llm)

    st.subheader("Answer")
    st.write(answer)
