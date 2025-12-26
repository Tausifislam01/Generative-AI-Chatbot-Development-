# RAG Assignment API (FastAPI + FAISS + Groq) + Streamlit UI

A Retrieval-Augmented Generation (RAG) service that can ingest multiple document types (PDF/DOCX/TXT/CSV/SQLite/images), chunk + embed content, store vectors in FAISS, and answer questions via an LLM (Groq). Includes a minimal Streamlit UI for upload / URL ingest / indexing / querying.

## Features

- **Document ingestion**
  - Upload files via `POST /ingest` (multipart)
  - Download + ingest via `POST /ingest_url` (JSON: `{ "url": "..." }`)
  - Optional raw upload via `POST /upload` (returns `doc_id` + stored filename)
- **Supported file types**
  - `.pdf` (text extraction + OCR fallback for scanned PDFs)
  - `.docx`, `.txt`
  - `.csv`
  - SQLite `.db`, `.sqlite`, `.sqlite3` (tables -> text)
  - Images `.png`, `.jpg`, `.jpeg` (OCR)
- **Chunking**
  - Clean text split into overlapping chunks (chunk size ~1000, overlap ~150)
- **Embeddings + Vector store**
  - SentenceTransformers embeddings
  - FAISS index persisted to disk (`data/index`)
- **Querying**
  - `POST /query` supports:
    - text question
    - optional `image_base64` OCR
    - optional multi-document filter (`doc_id` or `doc_ids`)
    - optional `return_context`
    - optional `use_langchain` (bonus)
  - Returns answer + source metadata (filename, page, chunk id, snippet, type/icon)
- **Dockerized**
  - `docker-compose.yml` runs API + UI
- **Streamlit UI**
  - Upload/Ingest URL/Build Index/Docs/Query tabs

---

## Project structure (typical)

```text
.
├── app/
│   ├── main.py
│   ├── ingest/
│   │   ├── loaders.py
│   │   ├── chunking.py
│   │   └── ocr.py
│   └── rag/
│       ├── embeddings.py
│       ├── vectorstore.py
│       └── llm_groq.py
├── streamlit_app.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── data/                  # created at runtime (uploads/chunks/index)
```


---

## Requirements

- Python 3.10+ (3.11 recommended)
- System deps for OCR:
  - **Tesseract OCR**
  - **Poppler** (needed for OCR’ing scanned PDFs via `pdf2image`)

On Windows, you may need to set:
- `TESSERACT_CMD` to the `tesseract.exe` path
- `POPPLER_PATH` to Poppler’s `bin` folder

---

## Environment setup

Create a `.env` file from the example:

```bash
cp .env.example .env
```

### Required
- `GROQ_API_KEY` – Groq API key for LLM calls.

### Optional
- `AUTO_REBUILD_INDEX` – automatically rebuild FAISS index after ingestion/delete (`true`/`false`)
- `MIN_RETRIEVAL_SCORE` – retrieval threshold (default `0.25`)
- `DEFAULT_USE_LANGCHAIN` – default for query-time `use_langchain` flag
- `OCR_PDF_MAX_PAGES` – max pages to OCR for scanned PDFs (default `10`)
- `TESSERACT_CMD` – explicit path to `tesseract` binary (Windows)
- `POPPLER_PATH` – explicit path to Poppler `bin` folder (Windows)

---

## Run with Docker (recommended)

1) Set env:

```bash
export GROQ_API_KEY="your_key_here"
```

2) Start:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- UI:  `http://localhost:8501`

---

## Run locally (without Docker)

1) Create venv + install deps:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

2) Configure env:

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY
```

3) Start API:

```bash
uvicorn app.main:app --reload --port 8000
```

4) Start UI (optional):

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## API Endpoints

### Health
`GET /health`

Returns config + supported types.

### Upload (bonus)
`POST /upload` (multipart `file`)

Stores file and returns `doc_id` + filenames.

### Ingest file (recommended for normal flow)
`POST /ingest` (multipart `file`)

- Saves file under `data/uploads/`
- Extracts text (and OCR for images / scanned PDFs)
- Chunks text and saves chunks under `data/chunks/{doc_id}.json`
- If `AUTO_REBUILD_INDEX=true`, rebuilds the FAISS index automatically.

Example:

```bash
curl -F "file=@sample.pdf" http://127.0.0.1:8000/ingest
```

### Ingest URL
`POST /ingest_url` (JSON)

```bash
curl -X POST http://127.0.0.1:8000/ingest_url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/some.pdf"}'
```

### Build / rebuild index
`POST /build_index`

If auto indexing is off, call this after ingestion:

```bash
curl -X POST http://127.0.0.1:8000/build_index
```

### List documents
`GET /documents`

Returns ingested docs (doc_id, filename, chunk count, etc.)

### Delete a document
`DELETE /documents/{doc_id}`

Deletes stored file + chunk JSON; if auto indexing is enabled it attempts to rebuild index.

### Query
`POST /query`

Payload:

```json
{
  "question": "What does the document say about payment terms?",
  "image_base64": "optional_base64_encoded_image",
  "top_k": 5,
  "min_score": 0.25,
  "use_langchain": false,
  "return_context": false,
  "doc_id": "optional_single_doc_id",
  "doc_ids": ["optional", "multiple", "doc_ids"]
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key points", "top_k":5, "min_score":0.25}'
```

Response includes:
- `answer` (string)
- `sources` (list with filename/doc_id/page/chunk/score/snippet/type/icon)
- `per_document_stats` (counts by doc)
- optional `context` if `return_context=true`

---

## Data persistence

Runtime data is stored under:

- `data/uploads/`  – stored uploaded files
- `data/chunks/`   – extracted chunks + metadata per document (`{doc_id}.json`)
- `data/index/`    – FAISS index + metadata (`faiss.index`, `meta.json`)

Docker mounts `./data:/app/data` so data persists across container restarts.

---

## Troubleshooting

### “FAISS index not loaded. Call /build_index first.”
If `AUTO_REBUILD_INDEX=false`, ingesting docs won’t rebuild the index automatically. Call `POST /build_index`.

### Scanned PDF OCR errors (Poppler)
If you see errors mentioning Poppler/pdfinfo/pdftoppm, install Poppler and ensure it’s on PATH, or set `POPPLER_PATH`.

### Tesseract not found
Install Tesseract and (Windows) set `TESSERACT_CMD` to the full path of `tesseract.exe`.

---

## Notes (for graders)

- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Vector store: FAISS (persisted on disk)
- LLM: Groq (default model: llama-3.3-70b-versatile)
- Bonus: Streamlit UI + Docker + LangChain option + multi-doc filtering + icons/metadata in sources
