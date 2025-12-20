from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from datetime import datetime
import uuid
import json
from typing import Any, Dict, List
from app.rag.llm_groq import GroqLLM
from app.ingest.loaders import load_txt
from app.ingest.chunking import chunk_text
from app.rag.embeddings import Embedder
from app.rag.vectorstore import FaissVectorStore

load_dotenv()

app = FastAPI(title="RAG Assignment API", version="0.4.0")

UPLOAD_DIR = Path("data/uploads")
CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


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


def build_context(sources: List[Dict[str, Any]]) -> str:
    blocks = []
    for s in sources:
        blocks.append(
            f"[SOURCE doc_id={s.get('doc_id')} chunk_id={s.get('chunk_id')} score={s.get('score'):.3f}]\n"
            f"{s.get('snippet')}\n"
        )
    return "\n".join(blocks).strip()


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

    chunks_count = None

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
                }
            )
    return records


@app.post("/build_index")
def build_index():
    
    records = _load_all_chunks()
    if not records:
        raise HTTPException(status_code=400, detail="No chunks found. Ingest at least one .txt file first.")

    texts = [r["text"] for r in records]
    embedder = get_embedder()
    vectors = embedder.embed_texts(texts)

    # Store metadata aligned with vectors
    meta = [
        {
            "doc_id": r["doc_id"],
            "original_filename": r["original_filename"],
            "stored_filename": r["stored_filename"],
            "chunk_id": r["chunk_id"],
            "start_char": r["start_char"],
            "end_char": r["end_char"],
            "text": r["text"],
            "snippet": (r["text"][:240] + "…") if len(r["text"]) > 240 else r["text"],
        }
        for r in records
    ]


    store = FaissVectorStore(INDEX_DIR)
    store.build(vectors, meta)

    # refresh global store
    global _store
    _store = store

    return {
        "indexed_chunks": len(records),
        "index_dir": str(INDEX_DIR),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }


@app.post("/query")
def query(payload: Dict[str, Any]):
    question = payload.get("question")
    top_k = int(payload.get("top_k", 5))
    doc_id_filter = payload.get("doc_id")  

    if not question or not isinstance(question, str):
        raise HTTPException(status_code=400, detail="Missing 'question' (string).")

    store = get_store()
    if store.index is None:
        raise HTTPException(status_code=400, detail="Index not built. Call /build_index first.")

    embedder = get_embedder()
    qvec = embedder.embed_query(question)

    # over-fetch then filter
    results = store.search(qvec, top_k=top_k * 5)

    sources = []
    context_blocks = []
    for score, m in results:
        if doc_id_filter and m.get("doc_id") != doc_id_filter:
            continue

        sources.append(
            {
                "score": score,
                "doc_id": m.get("doc_id"),
                "original_filename": m.get("original_filename"),
                "chunk_id": m.get("chunk_id"),
                "snippet": m.get("snippet"),
            }
        )

        context_blocks.append(
            f"[SOURCE doc_id={m.get('doc_id')} chunk_id={m.get('chunk_id')}]\n{m.get('text')}\n"
        )

        if len(sources) >= top_k:
            break

    context = "\n".join(context_blocks).strip()

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided sources. "
        "If the sources do not contain the answer, say you don't know. "
        "When you use information, cite it by referring to SOURCE tags (doc_id, chunk_id)."
    )

    user_prompt = f"Sources:\n{context}\n\nQuestion:\n{question}\n\nAnswer with citations."

    llm = GroqLLM()
    answer = llm.answer(system_prompt=system_prompt, user_prompt=user_prompt)

    return {"answer": answer, "sources": sources}

