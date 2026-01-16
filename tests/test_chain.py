from rag.chain import rag_chain
from rag.llm import get_llm  # your HF LLM wrapper

llm = get_llm()

question = "What's the main trend in banking as of 2025?"

answer = rag_chain(question, llm)

print(answer)
