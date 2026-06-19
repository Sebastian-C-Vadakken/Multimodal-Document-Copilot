# Multimodal Document Copilot

An AI-powered document copilot that allows users to upload PDF documents, automatically chunks and indexes them, and answers questions using a retrieval-augmented generation (RAG) approach.

## Architecture

- **Backend:** Python + FastAPI
- **AI / LLM:** Google Gemini API (`gemini-2.5-flash`)
- **Vector Database:** ChromaDB
- **Embeddings:** SentenceTransformer (`all-MiniLM-L6-v2`)
- **Frontend:** Vanilla HTML/CSS/JS (`index.htm`)

## Features

- **PDF Uploads:** Extract text content seamlessly from PDF documents using PyMuPDF.
- **RAG-based Question Answering:** Answer questions strictly based on the provided PDF context.
- **Modern Interface:** A dynamic, clean, and responsive web UI.
- **Local Embeddings:** Embeddings are generated locally using HuggingFace models for speed and privacy.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd multimodal-doc-copilot
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   cd backend
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the `backend` folder and add your Gemini API Key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the Backend:**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

5. **Run the Frontend:**
   Open `frontend/index.htm` in your web browser.

## How it works

1. The frontend lets you upload a PDF file.
2. The file is sent to the FastAPI backend where it is split into chunks.
3. These chunks are embedded and stored in a persistent local Chroma vector database.
4. When you ask a question, the relevant chunks are retrieved and passed to the Gemini API as context to generate an accurate response.
