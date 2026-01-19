from langchain_huggingface import HuggingFaceEmbeddings

#Hugging face MiniLM-L6-v2 model for embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )