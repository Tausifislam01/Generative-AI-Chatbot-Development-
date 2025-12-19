from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from datetime import datetime
import uuid

app = FastAPI(title="RAG Assignment API", version="0.2.0")


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
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

    return {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "size_bytes": len(content),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }
