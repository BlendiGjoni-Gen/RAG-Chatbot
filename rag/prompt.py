from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:

    system_prompt = """
You are a factual assistant answering questions using ONLY the provided context.

RULES:
- Use ONLY the information in the context.
- Do NOT use prior knowledge.
- Do NOT make up facts.
- If the answer is not contained in the context, say:
  "I don't know. The information is not available in the provided documents."
- Always include citations for every factual statement.
- Citations must include document name and page number.

FORMAT:
Answer:
<concise answer>

Citations:
- <document_name>, page <page_number>
    """

    human_prompt = """
Context:
{context}

Question:
{question}
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )