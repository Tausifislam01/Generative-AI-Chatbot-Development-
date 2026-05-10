import base64
import html
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = (os.getenv("STREAMLIT_API_URL") or os.getenv("API_BASE") or "http://127.0.0.1:8000").rstrip("/")
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "html", "htm", "csv", "db", "sqlite", "sqlite3", "png", "jpg", "jpeg"]

st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-xlight: #eef2ff;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-2: #475569;
    --text-3: #94a3b8;
    --success: #10b981;
    --success-bg: #ecfdf5;
    --warning: #f59e0b;
    --warning-bg: #fffbeb;
    --error: #ef4444;
    --error-bg: #fef2f2;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,.08), 0 4px 16px rgba(99,102,241,.06);
    --shadow-lg: 0 4px 24px rgba(99,102,241,.12);
}

.stApp { background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 50%, #f8fafc 100%); }
.block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
    box-shadow: 2px 0 16px rgba(0,0,0,.04);
}

h1,h2,h3 { font-family: 'Inter', sans-serif; color: var(--text); }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: var(--primary); margin-bottom: .5rem;
}
.page-hero {
    background: linear-gradient(120deg, #6366f1 0%, #818cf8 60%, #a5b4fc 100%);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg);
}
.page-hero h1 { color: white; font-size: 1.7rem; margin: 0 0 .3rem; }
.page-hero p { color: rgba(255,255,255,.85); margin: 0; font-size: .94rem; }
.badge {
    display: inline-block; padding: .2rem .55rem; border-radius: 999px;
    font-size: .72rem; font-weight: 600; margin-right: .3rem;
}
.badge-primary { background: var(--primary-xlight); color: var(--primary); }
.badge-success { background: var(--success-bg); color: var(--success); }
.badge-warn { background: var(--warning-bg); color: var(--warning); }
.badge-error { background: var(--error-bg); color: var(--error); }
.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: .8rem 1rem;
    margin-bottom: .6rem;
    box-shadow: var(--shadow);
}
.source-card .filename { font-weight: 600; color: var(--text); font-size: .9rem; }
.source-card .meta { color: var(--text-3); font-size: .78rem; margin: .2rem 0 .4rem; }
.source-card .snippet { color: var(--text-2); font-size: .85rem; line-height: 1.5; }
.stat-pill {
    display: inline-flex; align-items: center; gap: .4rem;
    background: var(--primary-xlight); color: var(--primary);
    border-radius: 999px; padding: .25rem .7rem;
    font-size: .8rem; font-weight: 600;
}
.auth-wrap {
    max-width: 440px; margin: 2rem auto;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 2.2rem; box-shadow: var(--shadow-lg);
}
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem;
    box-shadow: var(--shadow);
}
div[data-testid="stChatMessage"] { border-radius: var(--radius); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── helpers ────────────────────────────

def init_state():
    defaults = {
        "token": "", "user": None,
        "chat_history": [], "last_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def auth_headers() -> dict:
    t = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {t}"} if t else {}


def api(method: str, path: str, **kwargs) -> Any:
    resp = requests.request(method, f"{API_BASE}{path}", **kwargs)
    if resp.status_code >= 400:
        raise requests.HTTPError(resp.text, response=resp)
    return resp.json() if resp.content else {}


def safe_api(method, path, **kwargs):
    try:
        return api(method, path, **kwargs), None
    except Exception as e:
        msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            msg = e.response.text
        return None, msg


def health():
    try:
        return api("GET", "/health", timeout=4)
    except Exception:
        return None


def file_kind(name: str) -> str:
    ext = (name.rsplit(".", 1)[-1] if "." in name else "").lower()
    return {"pdf":"PDF","docx":"DOCX","txt":"TXT","html":"HTML","htm":"HTML",
            "csv":"CSV","db":"SQLite","sqlite":"SQLite","sqlite3":"SQLite",
            "png":"Image","jpg":"Image","jpeg":"Image"}.get(ext, "File")


def current_role() -> str:
    return str((st.session_state.get("user") or {}).get("role", "user"))


def can_manage():
    return current_role() in {"admin", "editor"}


def logout():
    st.session_state.token = ""
    st.session_state.user = None
    st.session_state.chat_history = []
    st.session_state.last_answer = None
    st.rerun()


# ─────────────────────────── auth page ──────────────────────────

def render_auth():
    h = health()
    status = "🟢 Service online" if h else "🔴 Service offline"

    st.markdown(f"""
    <div style="text-align:center; padding: 2rem 0 .5rem;">
        <div style="font-size:2.4rem; margin-bottom:.4rem;">🧠</div>
        <h1 style="font-size:1.9rem; color:#0f172a; margin:0 0 .3rem;">RAG Knowledge Chatbot</h1>
        <p style="color:#64748b; margin:0 0 .8rem; font-size:.95rem;">
            Source-grounded answers from your private document library
        </p>
        <span style="font-size:.8rem; color:#64748b;">{status}</span>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 1.4, 1])[1]
    with col:
        with st.container(border=True):
            mode = st.radio("Mode", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
            with st.form("auth_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password",
                                         placeholder="Min 8 characters" if mode == "Register" else "Your password")
                submitted = st.form_submit_button(
                    "Create Account" if mode == "Register" else "Sign In",
                    use_container_width=True, type="primary"
                )
            if submitted:
                if not email.strip() or not password:
                    st.error("Email and password are required.")
                elif mode == "Register":
                    data, err = safe_api("POST", "/auth/register",
                                         json={"email": email.strip(), "password": password}, timeout=20)
                    if err:
                        st.error(err)
                    else:
                        st.success("Account created — please sign in.")
                else:
                    data, err = safe_api("POST", "/auth/login",
                                         json={"email": email.strip(), "password": password}, timeout=20)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.token = data.get("access_token", "")
                        me, merr = safe_api("GET", "/auth/me", headers=auth_headers(), timeout=10)
                        st.session_state.user = me if me else {}
                        st.rerun()


# ─────────────────────────── sidebar ────────────────────────────

def render_sidebar(h):
    user = st.session_state.get("user") or {}
    with st.sidebar:
        st.markdown("""
        <div style="padding:.5rem 0 1rem;">
            <div style="font-size:1.5rem; font-weight:800; color:#6366f1;">🧠 RAG Chatbot</div>
            <div style="font-size:.78rem; color:#94a3b8; margin-top:.2rem;">Knowledge retrieval system</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.caption("SIGNED IN AS")
        st.markdown(f"**{user.get('email','—')}**")
        role = current_role()
        badge_cls = "badge-primary" if role == "admin" else ("badge-success" if role == "editor" else "badge-warn")
        st.markdown(f'<span class="badge {badge_cls}">{role.upper()}</span>', unsafe_allow_html=True)

        st.divider()
        st.caption("NAVIGATION")
        page = st.radio("Nav", ["💬 Chat", "📚 Knowledge Base", "📄 Documents", "⚙️ Settings"],
                        label_visibility="collapsed")
        st.divider()

        if h:
            st.markdown('<span class="badge badge-success">● API Online</span>', unsafe_allow_html=True)
            auto_idx = h.get("auto_rebuild_index", False)
            idx_cls = "badge-success" if auto_idx else "badge-warn"
            idx_lbl = "Auto-index ON" if auto_idx else "Auto-index OFF"
            st.markdown(f'<span class="badge {idx_cls}">{idx_lbl}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-error">● API Offline</span>', unsafe_allow_html=True)

        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            logout()

    return page


# ─────────────────────────── chat page ──────────────────────────

def render_chat(h, docs):
    st.markdown("""
    <div class="page-hero">
        <h1>💬 Chat</h1>
        <p>Ask questions grounded in your knowledge base — get cited answers.</p>
    </div>
    """, unsafe_allow_html=True)

    min_score_def = float(h.get("min_retrieval_score", 0.25)) if h else 0.25
    lc_def = bool(h.get("default_use_langchain", False)) if h else False

    label_to_id = {}
    doc_labels = []
    for d in docs:
        did = d.get("doc_id")
        fn = d.get("original_filename") or did
        if did:
            lbl = f"{file_kind(fn)} — {fn} [{did[:8]}]"
            label_to_id[lbl] = did
            doc_labels.append(lbl)

    chat_col, ctrl_col = st.columns([0.65, 0.35], gap="large")

    with ctrl_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">Retrieval Settings</div>', unsafe_allow_html=True)
            top_k = st.slider("Sources to retrieve", 1, 8, 5)
            min_score = st.slider("Min relevance score", 0.0, 1.0, min_score_def, 0.05)
            use_lc = st.toggle("Use LangChain generation", value=lc_def,
                               help="Uses LangChain + Groq pipeline instead of direct Groq SDK")
            show_ctx = st.toggle("Show retrieved context", value=False,
                                 help="Returns the raw text chunks sent to the LLM")
            st.markdown('<div class="card-title" style="margin-top:.8rem;">Document Filter</div>',
                        unsafe_allow_html=True)
            sel_docs = st.multiselect("Limit to documents", doc_labels,
                                      placeholder="All documents")
            st.markdown('<div class="card-title" style="margin-top:.8rem;">Image OCR</div>',
                        unsafe_allow_html=True)
            img_file = st.file_uploader("Attach image for OCR", type=["png", "jpg", "jpeg"],
                                        help="Image text will be added as an extra source")
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_answer = None
                st.rerun()

    with chat_col:
        if not docs:
            st.info("📭 No documents in the knowledge base yet. An admin can upload files from **Knowledge Base**.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask a question from your knowledge base…")
        if question:
            payload: dict = {
                "question": question.strip(),
                "top_k": int(top_k),
                "min_score": float(min_score),
                "use_langchain": bool(use_lc),
                "return_context": bool(show_ctx),
                "history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[-8:]
                ],
            }
            doc_ids = [label_to_id[l] for l in sel_docs if l in label_to_id]
            if doc_ids:
                payload["doc_ids"] = doc_ids
            if img_file is not None:
                payload["image_base64"] = base64.b64encode(img_file.getvalue()).decode()

            with st.spinner("Retrieving context and generating answer…"):
                result, err = safe_api("POST", "/query",
                                       json=payload, headers=auth_headers(), timeout=300)
            if err:
                st.error(err)
            else:
                answer = result.get("answer", "")
                st.session_state.chat_history.append({"role": "user", "content": question.strip()})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_answer = result
                st.rerun()

    # ── answer details ──
    latest = st.session_state.last_answer
    if not latest:
        return

    sources = latest.get("sources", []) or []
    per_doc = latest.get("per_document_stats", []) or []
    ctx = latest.get("context", "") or ""

    st.divider()
    if per_doc:
        st.markdown("#### 📊 Per-document stats")
        st.dataframe(pd.DataFrame(per_doc), use_container_width=True, hide_index=True)

    st.markdown("#### 🔗 Sources")
    if not sources:
        st.info("No sources were retrieved for the last answer.")
    for i, s in enumerate(sources, 1):
        fname = s.get("source") or s.get("original_filename") or "unknown"
        score = float(s.get("score") or 0)
        page = s.get("page") or "—"
        cid = s.get("chunk_id") if s.get("chunk_id") is not None else "—"
        snippet = s.get("snippet") or ""
        st.markdown(f"""
        <div class="source-card">
            <div class="filename">{i}. {html.escape(str(fname))}</div>
            <div class="meta">Page {html.escape(str(page))} &nbsp;·&nbsp; Chunk {html.escape(str(cid))} &nbsp;·&nbsp; Score {score:.3f}</div>
            <div class="snippet">{html.escape(str(snippet))}</div>
        </div>
        """, unsafe_allow_html=True)

    if show_ctx and ctx:
        st.markdown("#### 📋 Retrieved Context")
        st.code(ctx, language="text")


# ─────────────────────────── knowledge base page ────────────────

def render_knowledge(h):
    st.markdown("""
    <div class="page-hero">
        <h1>📚 Knowledge Base</h1>
        <p>Upload files, ingest URLs, and manage the FAISS search index.</p>
    </div>
    """, unsafe_allow_html=True)

    if not can_manage():
        st.warning("🔒 Admin or Editor role required to manage the knowledge base.")
        return

    auto_idx = h.get("auto_rebuild_index") if h else None

    # ── upload + url ──
    up_col, url_col = st.columns(2, gap="large")

    with up_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">Upload File</div>', unsafe_allow_html=True)
            st.caption(f"Supported: {', '.join(SUPPORTED_UPLOAD_TYPES)}")
            fup = st.file_uploader("Choose file", type=SUPPORTED_UPLOAD_TYPES, key="kb_uploader")
            if st.button("⬆️ Ingest File", use_container_width=True, type="primary", key="btn_ingest"):
                if fup is None:
                    st.error("Select a file first.")
                else:
                    with st.spinner("Uploading and processing…"):
                        res, err = safe_api("POST", "/ingest",
                                            files={"file": (fup.name, fup.getvalue())},
                                            headers=auth_headers(), timeout=300)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"✅ Ingested **{res.get('original_filename')}** — {res.get('chunks_count')} chunks")
                        st.json(res)
                        if auto_idx is False:
                            st.warning("Auto-index is OFF — rebuild the index below before querying.")

    with url_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">Ingest URL</div>', unsafe_allow_html=True)
            st.caption("Direct link to a PDF, DOCX, CSV, TXT, HTML, or image file.")
            url_in = st.text_input("Document URL", placeholder="https://example.com/report.pdf", key="url_input")
            if st.button("🌐 Ingest URL", use_container_width=True, type="primary", key="btn_ingest_url"):
                if not url_in.strip():
                    st.error("Enter a URL first.")
                else:
                    with st.spinner("Downloading and processing…"):
                        res, err = safe_api("POST", "/ingest_url",
                                            json={"url": url_in.strip()},
                                            headers=auth_headers(), timeout=300)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"✅ Ingested **{res.get('original_filename')}** — {res.get('chunks_count')} chunks")
                        st.json(res)
                        if auto_idx is False:
                            st.warning("Auto-index is OFF — rebuild the index below before querying.")

    st.divider()

    # ── index management ──
    idx_col, info_col = st.columns([0.4, 0.6], gap="large")
    with idx_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">Index Management</div>', unsafe_allow_html=True)
            st.caption("Rebuild the FAISS vector index from all ingested chunks.")
            if st.button("🔨 Build / Rebuild Index", use_container_width=True, type="primary", key="btn_build"):
                with st.spinner("Building FAISS index…"):
                    res, err = safe_api("POST", "/build_index", headers=auth_headers(), timeout=600)
                if err:
                    st.error(err)
                else:
                    st.success(f"✅ Index built — {res.get('indexed_chunks')} chunks indexed")
                    st.json(res)

    with info_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">Config from /health</div>', unsafe_allow_html=True)
            if h:
                cols = st.columns(2)
                cols[0].metric("Auto-rebuild Index", "ON" if h.get("auto_rebuild_index") else "OFF")
                cols[1].metric("Min Score", h.get("min_retrieval_score", "—"))
                cols2 = st.columns(2)
                cols2[0].metric("LangChain Default", "ON" if h.get("default_use_langchain") else "OFF")
                cols2[1].metric("OCR PDF Max Pages", h.get("ocr_pdf_max_pages", "—"))
                cols3 = st.columns(2)
                cols3[0].metric("CSV Max Rows", h.get("csv_max_rows", "—"))
                cols3[1].metric("SQLite Max Rows", h.get("sqlite_max_rows_per_table", "—"))
            else:
                st.warning("API not reachable.")


# ─────────────────────────── documents page ─────────────────────

def render_documents(docs, h):
    st.markdown("""
    <div class="page-hero">
        <h1>📄 Documents</h1>
        <p>Browse all ingested documents and remove them from the knowledge base.</p>
    </div>
    """, unsafe_allow_html=True)

    if not docs:
        st.info("📭 No documents ingested yet.")
        return

    total_chunks = sum(int(d.get("chunks_count") or 0) for d in docs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Documents", len(docs))
    c2.metric("Total Chunks", total_chunks)
    c3.metric("Avg Chunks/Doc", f"{total_chunks // max(len(docs), 1)}")

    st.markdown("---")

    rows = []
    for d in docs:
        fn = d.get("original_filename") or "unknown"
        rows.append({
            "Type": file_kind(fn),
            "Filename": fn,
            "Chunks": d.get("chunks_count", 0),
            "Document ID": d.get("doc_id", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not can_manage():
        return

    st.divider()
    st.markdown("#### 🗑️ Delete Document")
    auto_idx = h.get("auto_rebuild_index") if h else None

    label_map = {
        f"{file_kind(d.get('original_filename') or '')} — {d.get('original_filename') or d.get('doc_id')}": d.get("doc_id")
        for d in docs if d.get("doc_id")
    }
    choices = ["— Select a document —"] + list(label_map.keys())
    sel = st.selectbox("Document to delete", choices)

    if st.button("🗑️ Delete Selected Document", type="primary", use_container_width=True, key="btn_del"):
        if sel == choices[0]:
            st.error("Select a document first.")
        else:
            with st.spinner("Deleting…"):
                res, err = safe_api("DELETE", f"/documents/{label_map[sel]}",
                                    headers=auth_headers(), timeout=120)
            if err:
                st.error(err)
            else:
                st.success("✅ Document deleted.")
                st.json(res)
                if auto_idx is False:
                    st.warning("Auto-index is OFF — rebuild the index before querying.")
                st.rerun()


# ─────────────────────────── settings page ──────────────────────

def render_settings(h):
    st.markdown("""
    <div class="page-hero">
        <h1>⚙️ Settings</h1>
        <p>Session management and backend diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)

    user = st.session_state.get("user") or {}

    with st.container(border=True):
        st.markdown('<div class="card-title">Account</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Email", user.get("email", "—"))
        c2.metric("Role", current_role().upper())
        c3.metric("Active", "Yes" if user.get("is_active") else "No")
        st.caption(f"User ID: `{user.get('id', '—')}`")

    with st.container(border=True):
        st.markdown('<div class="card-title">Backend Health</div>', unsafe_allow_html=True)
        if h:
            st.success("✅ Backend is reachable and healthy.")
            st.json(h)
        else:
            st.error("❌ Backend is not reachable.")
        if st.button("🔄 Re-check health", key="btn_health"):
            st.rerun()

    with st.container(border=True):
        st.markdown('<div class="card-title">Session</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            if st.button("🔄 Refresh App", use_container_width=True):
                st.rerun()
        with b:
            if st.button("🚪 Sign Out", use_container_width=True, type="primary"):
                logout()

    with st.container(border=True):
        st.markdown('<div class="card-title">Supported File Types</div>', unsafe_allow_html=True)
        if h:
            exts = h.get("supported_exts", [])
            st.write(" · ".join(exts) if exts else "—")
        else:
            st.write(", ".join(SUPPORTED_UPLOAD_TYPES))


# ─────────────────────────── main ───────────────────────────────

if not st.session_state.user:
    render_auth()
    st.stop()

h_data = health()

try:
    docs_list = api("GET", "/documents", headers=auth_headers(), timeout=20) or []
    docs_list.sort(key=lambda d: (d.get("original_filename") or "").lower())
except Exception:
    docs_list = []

active_page = render_sidebar(h_data)

if active_page == "💬 Chat":
    render_chat(h_data, docs_list)
elif active_page == "📚 Knowledge Base":
    render_knowledge(h_data)
elif active_page == "📄 Documents":
    render_documents(docs_list, h_data)
else:
    render_settings(h_data)
