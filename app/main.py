from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pathlib import Path
from datetime import datetime
import uuid
import json
from typing import Any, Dict, List
from app.rag.llm_groq import GroqLLM
from app.ingest.loaders import load_txt, load_pdf, load_docx, load_csv, load_sqlite
from app.ingest.chunking import chunk_text
from app.rag.embeddings import Embedder
from app.rag.vectorstore import FaissVectorStore
import re
from app.ingest.ocr import ocr_image_file, ocr_pdf_file, ocr_image_bytes
import base64
import requests
import os

load_dotenv()

app = FastAPI(title="RAG Assignment API", version="0.5.0")

UPLOAD_DIR = Path("data/uploads")
CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

PAGE_MARKER_RE = re.compile(r"\[PAGE\s+(\d+)\]")

AUTO_REBUILD_INDEX = os.getenv("AUTO_REBUILD_INDEX", "true").strip().lower() in {"1", "true", "yes", "y"}
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))


def _friendly_ocr_error(e: Exception) -> HTTPException:
    msg = str(e)
    if "PDFInfoNotInstalledError" in msg or "Is poppler installed" in msg or "pdfinfo" in msg:
        return HTTPException(
            status_code=400,
            detail=(
                "This PDF appears to be scanned (or text extraction was insufficient), so OCR was attempted. "
                "OCR for PDFs requires Poppler (pdfinfo/pdftoppm). "
                "Install Poppler and ensure it's on PATH, or set POPPLER_PATH to the Poppler 'bin' folder. "
                "Example (Windows): setx POPPLER_PATH \"C:\\poppler\\Library\\bin\""
            ),
        )
    return HTTPException(status_code=400, detail=f"OCR failed: {e}")


def _build_page_starts(text: str) -> List[tuple[int, int]]:
    starts: List[tuple[int, int]] = []
    for m in PAGE_MARKER_RE.finditer(text or ""):
        try:
            page = int(m.group(1))
            starts.append((m.start(), page))
        except Exception:
            continue
    return starts


def _page_for_chunk(
    start_char: int,
    page_starts: List[tuple[int, int]],
    default: int | None = None,
) -> int | None:
    if not page_starts:
        return default

    current = default
    for idx, page in page_starts:
        if idx <= start_char:
            current = page
        else:
            break
    return current


_embedder: Embedder | None = None
_store: FaissVectorStore | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_store() -> FaissVectorStore:
    global _store
    if _store is None:
        _store = FaissVectorStore(INDEX_DIR)
        _store.load()
    return _store


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SUPPORTED_DOCX_EXTS = {".docx"}
SUPPORTED_CSV_EXTS = {".csv"}
SUPPORTED_DB_EXTS = {".db", ".sqlite", ".sqlite3"}


def _is_probably_scanned_pdf(extracted_text: str) -> bool:
    return len((extracted_text or "").strip()) < 50


def _load_all_chunks() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in CHUNKS_DIR.glob("*.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        doc_id = payload["doc_id"]
        original_filename = payload.get("original_filename")
        stored_filename = payload.get("stored_filename")

        for c in payload.get("chunks", []):
            records.append(
                {
                    "doc_id": doc_id,
                    "original_filename": original_filename,
                    "stored_filename": stored_filename,
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "start_char": c.get("start_char"),
                    "end_char": c.get("end_char"),
                    "page": c.get("page"),
                }
            )
    return records


def _rebuild_index() -> Dict[str, Any]:
    records = _load_all_chunks()
    if not records:
        raise HTTPException(
            status_code=400,
            detail="No chunks found. Ingest at least one supported file first.",
        )

    texts = [r["text"] for r in records]
    embedder = get_embedder()
    vectors = embedder.embed_texts(texts)

    meta = [
        {
            "doc_id": r["doc_id"],
            "original_filename": r["original_filename"],
            "stored_filename": r["stored_filename"],
            "chunk_id": r["chunk_id"],
            "start_char": r["start_char"],
            "end_char": r["end_char"],
            "page": r.get("page"),
            "text": r["text"],
            "snippet": (r["text"][:240] + "…") if len(r["text"]) > 240 else r["text"],
        }
        for r in records
    ]

    store = FaissVectorStore(INDEX_DIR)
    store.build(vectors, meta)

    global _store
    _store = store

    return {
        "indexed_chunks": len(records),
        "index_dir": str(INDEX_DIR),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "auto_rebuild_index": AUTO_REBUILD_INDEX,
        "min_retrieval_score": MIN_RETRIEVAL_SCORE,
    }



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

    text = None

    if suffix == ".txt":
        text = load_txt(stored_path)
    elif suffix == ".pdf":
        text = load_pdf(stored_path)
        if _is_probably_scanned_pdf(text):
            try:
                text = ocr_pdf_file(stored_path, max_pages=5)
            except Exception as e:
                raise _friendly_ocr_error(e)
    elif suffix in SUPPORTED_IMAGE_EXTS:
        text = ocr_image_file(stored_path)
    elif suffix in SUPPORTED_DOCX_EXTS:
        text = load_docx(stored_path)
    elif suffix in SUPPORTED_CSV_EXTS:
        text = load_csv(stored_path)
    elif suffix in SUPPORTED_DB_EXTS:
        text = load_sqlite(stored_path)

    if text is None or not str(text).strip():
        raise HTTPException(status_code=400, detail=f"No text could be extracted from '{file.filename}'")

    page_starts = _build_page_starts(text)
    default_page = 1 if suffix in SUPPORTED_IMAGE_EXTS else None

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
                "page": _page_for_chunk(c.start_char, page_starts, default=default_page),
            }
            for c in chunks
        ],
    }

    (CHUNKS_DIR / f"{doc_id}.json").write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resp = {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "size_bytes": len(content),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "chunks_count": len(chunks),
        "auto_rebuilt_index": False,
    }

    if AUTO_REBUILD_INDEX:
        try:
            _rebuild_index()
            resp["auto_rebuilt_index"] = True
        except Exception:
            resp["auto_rebuilt_index"] = False

    return resp


@app.post("/ingest_url")
def ingest_url(payload: Dict[str, Any] = Body(...)):
    url = payload.get("url")
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'url'")


    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}")

    filename = url.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Could not infer filename from URL")

    suffix = Path(filename).suffix.lower()
    doc_id = str(uuid.uuid4())
    stored_filename = f"{doc_id}{suffix}"
    stored_path = UPLOAD_DIR / stored_filename

    stored_path.write_bytes(resp.content)

    text = None

    if suffix == ".txt":
        text = load_txt(stored_path)
    elif suffix == ".pdf":
        text = load_pdf(stored_path)
        if _is_probably_scanned_pdf(text):
            try:
                text = ocr_pdf_file(stored_path, max_pages=5)
            except Exception as e:
                raise _friendly_ocr_error(e)
    elif suffix in SUPPORTED_IMAGE_EXTS:
        text = ocr_image_file(stored_path)
    elif suffix in SUPPORTED_DOCX_EXTS:
        text = load_docx(stored_path)
    elif suffix in SUPPORTED_CSV_EXTS:
        text = load_csv(stored_path)
    elif suffix in SUPPORTED_DB_EXTS:
        text = load_sqlite(stored_path)

    if text is None or not str(text).strip():
        raise HTTPException(status_code=400, detail=f"No text could be extracted from '{filename}'")

    page_starts = _build_page_starts(text)
    default_page = 1 if suffix in SUPPORTED_IMAGE_EXTS else None

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    chunks_payload = {
        "doc_id": doc_id,
        "original_filename": filename,
        "stored_filename": stored_filename,
        "chunking": {"chunk_size": 1000, "overlap": 150},
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "page": _page_for_chunk(c.start_char, page_starts, default=default_page),
            }
            for c in chunks
        ],
    }

    (CHUNKS_DIR / f"{doc_id}.json").write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = {
        "doc_id": doc_id,
        "original_filename": filename,
        "stored_filename": stored_filename,
        "size_bytes": len(resp.content),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "chunks_count": len(chunks),
        "auto_rebuilt_index": False,
    }

    if AUTO_REBUILD_INDEX:
        try:
            _rebuild_index()
            out["auto_rebuilt_index"] = True
        except Exception:
            out["auto_rebuilt_index"] = False

    return out


@app.get("/documents")
def list_documents():
    docs = []
    for p in CHUNKS_DIR.glob("*.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        docs.append(
            {
                "doc_id": payload.get("doc_id"),
                "original_filename": payload.get("original_filename"),
                "stored_filename": payload.get("stored_filename"),
                "chunks_count": len(payload.get("chunks", [])),
            }
        )
    return docs


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    chunk_file = CHUNKS_DIR / f"{doc_id}.json"
    if not chunk_file.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    payload = json.loads(chunk_file.read_text(encoding="utf-8"))
    stored_filename = payload.get("stored_filename")

    if stored_filename:
        upload_file = UPLOAD_DIR / stored_filename
        if upload_file.exists():
            upload_file.unlink()

    chunk_file.unlink()

    return {
        "deleted_doc_id": doc_id,
        "note": "Document deleted. Please rebuild index to reflect deletion.",
        "auto_rebuilt_index": False,
    }



@app.post("/build_index")
def build_index():
    return _rebuild_index()


def _source_name(m: Dict[str, Any]) -> str:

    return (m.get("original_filename") or m.get("stored_filename") or m.get("doc_id") or "unknown").strip()


@app.post("/query")
def query(payload: Dict[str, Any]):
    question = payload.get("question")
    image_b64 = payload.get("image_base64")
    top_k = int(payload.get("top_k", 5))
    doc_id_filter = payload.get("doc_id")
    min_score = float(payload.get("min_score", MIN_RETRIEVAL_SCORE))

    if not question or not isinstance(question, str):
        raise HTTPException(status_code=400, detail="Missing 'question' (string).")

    top_k = max(1, min(top_k, 8))

    query_for_search = question.strip()
    question_for_llm = query_for_search

    ocr_text = ""
    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64)
            ocr_text = (ocr_image_bytes(img_bytes) or "").strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")

    store = get_store()
    if store.index is None:
        raise HTTPException(status_code=400, detail="Index not built. Call /build_index first.")

    embedder = get_embedder()
    qvec = embedder.embed_query(query_for_search)
    results = store.search(qvec, top_k=top_k * 10)

    sources: List[Dict[str, Any]] = []
    context_blocks: List[str] = []

    # Include OCR (if provided) as a first-class "source"
    if ocr_text:
        sources.append(
            {
                "score": 1.0,
                "source": "image_base64 (OCR)",
                "chunk_id": "ocr",
                "page": 1,
                "snippet": (ocr_text[:240] + "…") if len(ocr_text) > 240 else ocr_text,
            }
        )
        context_blocks.append(f"[SOURCE file=image_base64 (OCR) chunk_id=ocr page=1]\n{ocr_text}\n")
        question_for_llm = f"{question_for_llm}\n\n[IMAGE OCR INCLUDED AS SOURCE: file=image_base64 (OCR)]"

    retrieved_count = 0
    for score, m in results:
        if score < min_score:
            continue
        if doc_id_filter and m.get("doc_id") != doc_id_filter:
            continue

        filename = _source_name(m)

        sources.append(
            {
                "score": float(score),
                "source": filename,
                "chunk_id": m.get("chunk_id"),
                "page": m.get("page"),
                "snippet": m.get("snippet"),
            }
        )

        context_blocks.append(
            f"[SOURCE file={filename} chunk_id={m.get('chunk_id')} page={m.get('page')}]\n{m.get('text')}\n"
        )

        retrieved_count += 1
        if retrieved_count >= top_k:
            break

    context = "\n".join(context_blocks).strip()

    if not context:
        return {
            "answer": "I don't know. No relevant sources were retrieved.",
            "sources": [],
        }

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided sources. "
        "If the sources do not contain the answer, say you don't know. "
        "When you use information, cite it by referring to SOURCE tags (file, chunk_id, page)."
    )

    user_prompt = f"Sources:\n{context}\n\nQuestion:\n{question_for_llm}\n\nAnswer with citations."

    llm = GroqLLM()
    answer = llm.answer(system_prompt=system_prompt, user_prompt=user_prompt)

    return {"answer": answer, "sources": sources}
