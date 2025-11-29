import os
import traceback
from uuid import uuid4
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

# ================== ENV + GEMINI SETUP ==================
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment. Check your .env file.")

genai.configure(api_key=api_key)

# ================== FASTAPI APP ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where PDFs will be saved
UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================== VECTOR DB (CHROMA) ==================
VECTOR_DIR = os.path.join("data", "vector_store")
os.makedirs(VECTOR_DIR, exist_ok=True)

# Persistent Chroma client (data saved on disk)
chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)

# Local embedding function (runs on your machine, no extra API)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Collection where all chunks go
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn,
)

# ================== Pydantic models ==================
class QuestionRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 5  # how many chunks to retrieve


# ================== HEALTH ==================
@app.get("/health")
def health():
    return {"status": "ok"}


# ================== GEMINI TEST ==================
@app.get("/test-gemini")
def test_gemini():
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say a very short greeting.")

        text = getattr(response, "text", None)
        if not text:
            if hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts
                text = "".join(p.text for p in parts if hasattr(p, "text"))
            else:
                text = str(response)

        return {"response": text}

    except Exception as e:
        print("Error in /test-gemini:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
        )


# ================== HELPER: PDF CHUNKING ==================
def extract_pdf_chunks(pdf_path: str, doc_id: str) -> List[dict]:
    """
    Open a PDF file, read each page, split into paragraphs,
    and return a list of chunk dicts:
    { doc_id, page, chunk_id, text }
    """
    doc = fitz.open(pdf_path)
    chunks: List[dict] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")  # plain text extraction
        page_num = page_index + 1

        # Very simple paragraph splitting: split on blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            chunks.append(
                {
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_id": f"{page_num}-{i}",
                    "text": para,
                }
            )

    doc.close()
    return chunks


# ================== HELPER: GEMINI ANSWER ==================
def answer_with_gemini(question: str, context_chunks: List[dict]) -> str:
    """
    Use Gemini to answer a question based on retrieved chunks.
    Each chunk has: text, page, chunk_id.
    """
    if not context_chunks:
        return "I couldn't find any relevant content in the document for this question."

    context_text = ""
    for c in context_chunks:
        context_text += (
            f"[page {c['page']}, chunk {c['chunk_id']}]\n{c['text']}\n\n"
        )

    prompt = f"""
You are a strict PDF question-answering assistant.

Rules:
- Use ONLY the context below.
- Do NOT say that the context is incomplete unless it truly is.
- Answer in 3–5 sentences maximum. Be extremely concise.
- If the user asks for a summary, keep it under 50 words.
- Always include citations like (page X, chunk Y) after each key fact.

Context:
{context_text}

Question: {question}
"""


    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    text = getattr(response, "text", None)
    if not text:
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            text = "".join(p.text for p in parts if hasattr(p, "text"))
        else:
            text = str(response)

    return text


# ================== PDF UPLOAD + INDEX ==================
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Basic type check
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 2. Generate a unique document ID
    doc_id = str(uuid4())

    # 3. Save the uploaded file
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        print("Error saving PDF:", e)
        raise HTTPException(status_code=500, detail="Failed to save PDF file.")

    # 4. Parse and chunk the PDF
    try:
        chunks = extract_pdf_chunks(save_path, doc_id)
    except Exception as e:
        print("Error parsing PDF:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

    # 5. Index chunks in Chroma
    if chunks:
        ids = [f"{c['doc_id']}-{c['chunk_id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"doc_id": c["doc_id"], "page": c["page"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    pages = len(set(c["page"] for c in chunks)) if chunks else 0
    return {
        "doc_id": doc_id,
        "pages": pages,
        "chunks_indexed": len(chunks),
    }


# ================== ASK QUESTION ==================
@app.post("/ask_question")
def ask_question(payload: QuestionRequest):
    try:
        # 1. Retrieve relevant chunks from Chroma for this doc_id
        results = collection.query(
            query_texts=[payload.question],
            n_results=payload.top_k,
            where={"doc_id": payload.doc_id},
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found for this document/question.",
            )

        # Build chunk list for Gemini
        context_chunks = []
        for text, meta in zip(docs, metas):
            context_chunks.append(
                {
                    "text": text,
                    "page": meta.get("page", "?"),
                    "chunk_id": meta.get("chunk_id", "?"),
                }
            )

        # 2. Ask Gemini using those chunks
        answer = answer_with_gemini(payload.question, context_chunks)

        return {
            "answer": answer,
            "sources": context_chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        print("Error in ask_question:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
