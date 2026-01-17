from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        stop_sequences=["Answer:", "\n\n", "DONE", "<|eot_id|>"],
        temperature=0.1,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)
