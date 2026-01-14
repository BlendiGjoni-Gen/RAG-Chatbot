from rag.loader import load_documents
from rag.chunking import chunk_docs

DOCS_PATH = "data/documents"

# Step 1: Load documents
docs = load_documents(DOCS_PATH)
print(f"Loaded {len(docs)} documents")

# Step 2: Chunk documents
chunks = chunk_docs(docs)
print(f"Created {len(chunks)} chunks")

# Step 3: Inspect a sample
sample = chunks[0]

print("\n--- SAMPLE CHUNK ---")
print(sample.page_content[:500])  # first 500 chars
print("\n--- METADATA ---")
print(sample.metadata)
