import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import streamlit as st

def get_llm():
    # Attempt to get token from Streamlit secrets or Env vars
    hf_token = st.secrets["HF_TOKEN"]["token"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    if not hf_token:
        st.error("HF_TOKEN not found! Please set it in Streamlit Secrets.")
        return None

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",  # Explicitly define the task
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token,  # Ensure this is passed
    )
    return ChatHuggingFace(llm=llm)



