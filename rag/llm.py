import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.language_models import BaseChatModel

load_dotenv()

def get_llm() -> BaseChatModel:
    """
    Hugging Face hosted LLM (Inference API)
    """

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("Set HUGGINGFACEHUB_API_TOKEN environment variable")

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="conversational",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=llm)
