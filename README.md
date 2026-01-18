# 📚 RAG Chatbot

This is a **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit**.  
It can answer questions using a set of documents (PDFs and Markdown). The bot will **only use the info in your documents**, so it won’t make stuff up.  

---

## Features

- Ask questions based on uploaded PDFs or Markdown files.  
- Answers come only from your documents.  
- Uses a vectorstore (**FAISS**) to find relevant info.  
- Reranks documents to give better answers.  
- Uses **HuggingFace LLaMA 3.1 8B** model for generating answers.  
- Easy web interface with **Streamlit**.  

---

## How to Run

1. **Install dependencies:**
 
pip install -r requirements.txt
Add your HuggingFace API key to a .env file:

HUGGINGFACE_API_KEY=your_api_key_here

Prepare your documents in a folder, e.g., data/docs/.

Run the chatbot:

streamlit run app/streamlit_app.py
Go to http://localhost:8501 in your browser, type a question, and click Ask.

Notes

The chatbot will only answer based on your documents. If it can’t find the answer, it will say "I don't know."

Make sure you create the vectorstore before using the chatbot.