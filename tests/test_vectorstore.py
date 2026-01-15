from rag.loader import load_documents
from rag.chunking import chunk_docs
from rag.vectorstore import create_vectorstore


def main():
    docs = load_documents("data/documents")
    chunks = chunk_docs(docs)

    vectorstore = create_vectorstore(chunks)

    print(f"Vectorstore created with {len(chunks)} chunks")


if __name__ == "__main__":
    main()
