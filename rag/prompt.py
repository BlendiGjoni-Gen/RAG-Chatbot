from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:
    system_prompt = """You are a focused research assistant.
    RULES:
    1. Use ONLY the provided context. 
    2. If the answer is not in the context, say "I don't know."
    3. Format citations as [Filename], page [X].
    4. STOP after the answer.

    Context: {context}
    Question: {question}
    DONE"""

    human_prompt = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])