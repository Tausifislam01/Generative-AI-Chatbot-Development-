# RAG Assignment API (FastAPI + FAISS + Groq) + Streamlit UI

A Retrieval-Augmented Generation (RAG) service that can ingest multiple document types (PDF/DOCX/TXT/CSV/SQLite/images), chunk + embed content, store vectors in FAISS, and answer questions via an LLM (Groq). Includes a minimal Streamlit UI for upload / URL ingest / indexing / querying.

---

## Features

- **Document ingestion**
  - Upload files via `POST /ingest` (multipart)
  - Download + ingest via `POST /ingest_url`
  - Optional raw upload via `POST /upload`
- **Supported file types**
  - PDF (with OCR fallback)
  - DOCX, TXT
  - CSV
  - SQLite databases
  - Images (PNG/JPG OCR)
- **Chunking + Embeddings**
  - Overlapping chunks
  - SentenceTransformers embeddings
  - FAISS vector store (persisted)
- **Querying**
  - `/query` endpoint with:
    - question
    - optional image OCR
    - optional multi-document filtering
    - optional context return
- **Dockerized backend + Streamlit UI**

---

## Project Structure

```text
.
├── app/
├── streamlit_app.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── samples/
└── data/
```

---

## Sample Files

The repository includes a `samples/` directory with demo documents for grading and testing:

- `samples/sample_invoice.pdf`
- `samples/sample_policy.docx`
- `samples/sample_employees.csv`
- `samples/sample_shop.sqlite`

These files ensure answers can clearly reference **filename + page / table / row**, as required.

---

## Sample Queries (Demo / Grading)

### PDF — `sample_invoice.pdf`
- What are the payment terms for this invoice?
- What is the invoice due date?
- What is the total amount due?
- What late fee applies to overdue balances?
- What are the remittance instructions?
- On which page are the remittance instructions listed?

### DOCX — `sample_policy.docx`
- What is the purpose of the IT Access Policy?
- When does this policy become effective?
- What is the minimum password length requirement?
- Is multi-factor authentication required?
- Who should employees contact for access requests?

### CSV — `sample_employees.csv`
- Which employees are located in Dhaka?
- Who joined the company most recently?
- How many employees work in each department?

### SQLite — `sample_shop.sqlite`
- Which orders are currently pending?
- What are the payment terms for pending orders?
- What is the total order amount for “Example Buyer Ltd.”?
- Which customers have more than one order?

### Cross-document (RAG capability)
- Is “Net 30” mentioned in both the invoice and the database?
- Which document mentions late fees and which mentions payment terms?

---

## Environment Setup

```bash
cp .env.example .env
```

### Required
- `GROQ_API_KEY`

### Optional
- `AUTO_REBUILD_INDEX`
- `MIN_RETRIEVAL_SCORE`
- `DEFAULT_USE_LANGCHAIN`
- `OCR_PDF_MAX_PAGES`
- `TESSERACT_CMD`
- `POPPLER_PATH`

---

## Run with Docker (Recommended)

```bash
export GROQ_API_KEY="your_key"
docker compose up --build
```

- API: http://localhost:8000
- UI:  http://localhost:8501

---

## API Contract (Assignment Requirement)

The `/query` API returns:
- `answer` – final answer
- `sources` – filename, page, chunk id, snippet
- `context` – always present (empty unless requested)

---

## Notes for Graders

- Vector store: FAISS
- Embeddings: SentenceTransformers
- LLM: Groq
- OCR: Tesseract + Poppler
- Bonus: Streamlit UI, Docker, LangChain option, multi-document filtering

---

## Assignment Mapping (Requirement → Implementation)

This section maps each assignment requirement directly to the implemented feature or endpoint.

### 1. Document Upload & Ingestion
**Requirement:** Upload documents for processing  
**Implementation:**
- `POST /ingest` – upload and ingest files
- `POST /ingest_url` – download and ingest from URL
- `POST /upload` – bonus raw upload returning `doc_id`

### 2. Supported File Types
**Requirement:** Handle multiple document formats  
**Implementation:**
- PDF (text + OCR fallback)
- DOCX, TXT
- CSV
- SQLite databases (`.db`, `.sqlite`)
- Images (`.png`, `.jpg`, `.jpeg`) with OCR

### 3. Text Chunking
**Requirement:** Split documents into manageable chunks  
**Implementation:**
- Overlapping chunking (≈1000 tokens, overlap ≈150)
- Stored per-document under `data/chunks/{doc_id}.json`

### 4. Embeddings & Vector Store
**Requirement:** Convert text to embeddings and store for retrieval  
**Implementation:**
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS vector store persisted on disk (`data/index`)

### 5. Retrieval-Augmented Generation (RAG)
**Requirement:** Retrieve relevant chunks and generate answers  
**Implementation:**
- Similarity search via FAISS
- Context construction with source tags
- LLM response using Groq (LangChain optional)

### 6. Question Answering API
**Requirement:** API accepts a question and returns an answer  
**Implementation:**
- `POST /query`
- Supports optional image OCR (`image_base64`)
- Supports multi-document filtering
- Returns:
  - `answer`
  - `sources` (filename, page, chunk id, snippet)
  - `context` (always present; empty unless requested)

### 7. Image-based Questions (OCR)
**Requirement:** Answer questions from images  
**Implementation:**
- Tesseract OCR for images and scanned PDFs
- OCR text included as a retrievable source

### 8. API Response Transparency
**Requirement:** Include source information with answers  
**Implementation:**
- Source metadata includes filename, page number, chunk id, score, and snippet

### 9. Submission Artifacts
**Requirement:** GitHub repo / ZIP with code, README, env example  
**Implementation:**
- Full source code included
- `README.md` with setup, usage, and demos
- `.env.example` provided
- Sample files included in `samples/` directory

### 10. Bonus Features
**Implementation:**
- Streamlit UI
- Docker + docker-compose
- LangChain optional path
- Multi-document filtering
- Persistent vector index
- Source icons and metadata

---

## User Manual

This section is a practical guide for **running and using** the system (API + UI) end-to-end.

### Quick Start (Recommended Flow)

1) Start the system (Docker recommended):
```bash
export GROQ_API_KEY="your_key_here"
docker compose up --build
```

2) Open the UI:
- Streamlit UI: `http://localhost:8501`

3) In the UI:
- Upload a sample file (or ingest a URL)
- If Auto Indexing is OFF, click **Build Index**
- Ask a question in the **Query** tab and review **Sources** (and Context if enabled)

---

## API User Manual

### Base URLs
- API (default): `http://localhost:8000`
- UI (default):  `http://localhost:8501`

### 1) Check health
```bash
curl http://127.0.0.1:8000/health
```
Use this to confirm the service is running and to see config like `auto_rebuild_index` and `min_retrieval_score`.

### 2) Ingest a file (PDF/DOCX/TXT/CSV/SQLite/Image)
```bash
curl -F "file=@samples/sample_invoice.pdf" http://127.0.0.1:8000/ingest
```

If `AUTO_REBUILD_INDEX=true`, the document becomes queryable immediately.
If `AUTO_REBUILD_INDEX=false`, build the index next.

### 3) Ingest from URL
```bash
curl -X POST http://127.0.0.1:8000/ingest_url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/some.pdf"}'
```

### 4) Build / rebuild the FAISS index
Only required when auto-indexing is OFF (or if you want to force a rebuild):
```bash
curl -X POST http://127.0.0.1:8000/build_index
```

### 5) List ingested documents
```bash
curl http://127.0.0.1:8000/documents
```

### 6) Ask a question (RAG)
Minimal:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the payment terms?", "top_k":5, "min_score":0.25}'
```

With context returned:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the payment terms?", "top_k":5, "min_score":0.25, "return_context":true}'
```

With multi-document filtering:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key points", "doc_ids":["<doc_id_1>","<doc_id_2>"], "top_k":5}'
```

With image OCR:
```bash
# (Example) Create base64 from an image:
#   Linux/macOS: base64 -i image.png | tr -d '\n'
#   Windows PowerShell: [Convert]::ToBase64String([IO.File]::ReadAllBytes("image.png"))

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the image say about payment terms?","image_base64":"<BASE64_STRING>","top_k":5,"return_context":true}'
```

### 7) Understand the `/query` response
The response is designed to satisfy the assignment requirement:
- `answer`: final answer text
- `sources`: list of source objects with `source/original_filename`, `page`, `chunk_id`, `score`, and `snippet`
- `context`: retrieval context (always present; empty unless `return_context=true`)

### 8) Delete a document
```bash
curl -X DELETE http://127.0.0.1:8000/documents/<doc_id>
```
If auto-indexing is OFF, rebuild the index afterwards.

---

## UI User Manual (Streamlit)

Open the UI at `http://localhost:8501`.

### Sidebar
- **API Base URL**: change if your API runs elsewhere (default `http://127.0.0.1:8000`)
- **Check API Health**: verifies connectivity and shows current config
- **Use LangChain**: toggles the optional LangChain path (bonus)
- **Return context**: includes retrieved context in API response (useful for grading)

### Tabs

#### 1) Upload File
- Choose a file (PDF/DOCX/TXT/CSV/SQLite/Image)
- Click **Ingest File**
- If Auto Indexing is OFF, go to **Build Index** tab next

#### 2) Ingest URL
- Paste a direct link to a file (PDF/DOCX/etc.)
- Click **Ingest URL**
- If Auto Indexing is OFF, build the index next

#### 3) Build Index
- Click **Build Index** to create/rebuild the FAISS index
- Required if Auto Indexing is OFF

#### 4) Docs
- View ingested documents and their `doc_id`
- Delete documents if needed

#### 5) Query
- Enter a question
- Optionally select document filters (multi-doc)
- Optionally upload an image for OCR
- Click **Ask**
- Review:
  - **Answer**
  - **Per-document stats**
  - **Sources** (with page + snippet)
  - **Context** (if enabled)

---

## User-Facing Troubleshooting (Common)

- **API not reachable from UI**: confirm API is running and the API Base URL is correct.
- **“FAISS index not loaded. Call /build_index first.”**: build index or enable `AUTO_REBUILD_INDEX=true`.
- **Scanned PDF OCR errors**: install Poppler and set `POPPLER_PATH` (Windows) or ensure Poppler is on PATH.
- **Tesseract not found**: install Tesseract and set `TESSERACT_CMD` (Windows) if needed.
