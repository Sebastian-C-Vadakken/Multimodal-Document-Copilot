import os
import traceback
from uuid import uuid4
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF

# ----- ENV + GEMINI SETUP -----
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment. Check your .env file.")

genai.configure(api_key=api_key)

# ----- FASTAPI APP -----
app = FastAPI()

# Where PDFs will be saved
UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----- HEALTH CHECK -----
@app.get("/health")
def health():
    return {"status": "ok"}


# ----- GEMINI TEST ENDPOINT -----
@app.get("/test-gemini")
def test_gemini():
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
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


# ----- PDF CHUNKING HELPER -----
def extract_pdf_chunks(pdf_path: str, doc_id: str) -> List[dict]:
    """
    Open a PDF file, read each page, split into paragraphs,
    and return a list of chunk dicts:
    { doc_id, page, chunk_id, text }
    """
    doc = fitz.open(pdf_path)
    chunks = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")  # plain text extraction
        page_num = page_index + 1

        # Very simple paragraph splitting: split on blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            chunks.append({
                "doc_id": doc_id,
                "page": page_num,
                "chunk_id": f"{page_num}-{i}",
                "text": para,
            })

    doc.close()
    return chunks


# ----- PDF UPLOAD ENDPOINT -----
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

    # 5. For now we just return stats. Later we'll store chunks in a vector DB.
    pages = len(set(c["page"] for c in chunks))
    return {
        "doc_id": doc_id,
        "pages": pages,
        "chunks_indexed": len(chunks),
    }
