# 📚 RAG Chatbot

This is a **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit**.  
It can answer questions using a set of documents (PDFs and Markdown). The bot will **only use the info in your documents**, so it won’t make stuff up.  

---

## Features

- Ask questions based on uploaded PDFs or Markdown files.  
- Answers come only from your documents.  
- Uses a vectorstore (**FAISS**) to find relevant info. 
- Uses **HuggingFace LLaMA 3.1 8B** model for generating answers.  
- Easy web interface with **Streamlit**, hosted on **Hugging Face Streamlit Space**.
- Includes **tests** on all phases of the pipeline.
- _Reranks documents to give better answers._ 
- _Has **guardrails** to prevent instruction ignoring_

---

## Questions to test

- What are some banking trends as of 2024?
- What do outlooks say about banking in recent years?
- What are some interesting facts about finance over these years?

---
## How to Run (URL Demo)

On any browser, go to https://blendigjoni-ragchatbot.streamlit.app/

Ask your questions!

## How to Run (LOCALLY)

### 1. **Create and activate venv**

`python -m venv .venv`

`venv\Scripts\activate`

### 2. **Install dependencies:**
 
`pip install -r requirements.txt`

### 3. Add your HuggingFace API key to a .env file:
`HUGGINGFACE_API_KEY=your_api_key_here`

### 4. Prepare your documents in a folder, e.g., data/docs/.

**_Run the chatbot:_**

`streamlit run app/streamlit_app.py`

Go to http://localhost:8501 in your browser, type a question, and click Ask.

### Notes

The chatbot will only answer based on your documents. If it can’t find the answer, it will say "I don't know."

Make sure you create the vectorstore before using the chatbot locally.

