from rag.retriever import get_retriever


def main():
    retriever = get_retriever(k=3)

    query = "growth in millionaire population"
    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:300])
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))


if __name__ == "__main__":
    main()
