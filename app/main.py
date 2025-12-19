from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from datetime import datetime
import uuid
import json

from app.ingest.loaders import load_txt
from app.ingest.chunking import chunk_text

app = FastAPI(title="RAG Assignment API", version="0.3.0")

UPLOAD_DIR = Path("data/uploads")
CHUNKS_DIR = Path("data/chunks")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document and save it to disk.
    If it's a .txt file, also:
      - load text
      - chunk it
      - save chunks JSON under data/chunks/{doc_id}.json
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    doc_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix.lower()
    stored_filename = f"{doc_id}{suffix}"
    stored_path = UPLOAD_DIR / stored_filename

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    stored_path.write_bytes(content)

    chunks_count = None

    # TXT-only for this step
    if suffix == ".txt":
        text = load_txt(stored_path)
        chunks = chunk_text(text, chunk_size=1000, overlap=150)

        chunks_payload = {
            "doc_id": doc_id,
            "original_filename": file.filename,
            "stored_filename": stored_filename,
            "chunking": {"chunk_size": 1000, "overlap": 150},
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
                for c in chunks
            ],
        }

        (CHUNKS_DIR / f"{doc_id}.json").write_text(
            json.dumps(chunks_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        chunks_count = len(chunks)

    return {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "size_bytes": len(content),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "chunks_count": chunks_count,
    }
