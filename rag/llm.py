import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        max_new_tokens=512,
        provider="huggingface",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        stop_sequences=["DONE"]
    )
    return ChatHuggingFace(llm=llm)
