import os
import base64
import requests
import streamlit as st
import pandas as pd
from typing import List

API_BASE_DEFAULT = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="RAG Assignment UI", layout="wide")
st.title("RAG Assignment UI")

with st.sidebar:
    st.subheader("Settings")
    api_base = st.text_input("API Base URL", value=API_BASE_DEFAULT).rstrip("/")
    st.caption("Example: http://127.0.0.1:8000")
    st.divider()

    auto_indexing = None
    min_score_default = None
    default_use_langchain = None

    if st.button("Check API Health"):
        try:
            r = requests.get(f"{api_base}/health", timeout=10)
            if r.status_code == 200:
                data = r.json()
                st.success(data)
                auto_indexing = data.get("auto_rebuild_index")
                min_score_default = data.get("min_retrieval_score")
                default_use_langchain = data.get("default_use_langchain")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

    try:
        r = requests.get(f"{api_base}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            auto_indexing = data.get("auto_rebuild_index")
            min_score_default = data.get("min_retrieval_score")
            default_use_langchain = data.get("default_use_langchain")
    except Exception:
        pass

    st.divider()
    st.subheader("Indexing Status")
    if auto_indexing is True:
        st.success("Auto indexing: ON ✅")
        st.caption("Uploads/URL ingests are queryable immediately.")
    elif auto_indexing is False:
        st.warning("Auto indexing: OFF ⚠️")
        st.caption("After ingesting files, you must run Build Index.")
    else:
        st.info("Auto indexing: unknown")
        st.caption("API not reachable or /health missing config fields.")

    st.divider()
    st.subheader("Answer Options")
    use_langchain_ui = st.checkbox(
        "Use LangChain (bonus)",
        value=bool(default_use_langchain) if default_use_langchain is not None else False,
    )
    return_context_ui = st.checkbox("Return context in response", value=False)

API_BASE = api_base


def icon_for(name: str) -> str:
    ext = (name.split(".")[-1] if "." in name else "").lower()
    return {
        "pdf": "📄",
        "docx": "📝",
        "txt": "📃",
        "csv": "📊",
        "db": "🗄️",
        "sqlite": "🗄️",
        "sqlite3": "🗄️",
        "png": "🖼️",
        "jpg": "🖼️",
        "jpeg": "🖼️",
    }.get(ext, "📁")


def fetch_documents() -> list[dict]:
    r = requests.get(f"{API_BASE}/documents", timeout=30)
    r.raise_for_status()
    docs = r.json() or []
    docs.sort(key=lambda d: (d.get("original_filename") or "").lower())
    return docs


tabs = st.tabs(["Upload File", "Ingest URL", "Build Index", "Docs", "Query"])

with tabs[0]:
    st.subheader("Upload File")
    file = st.file_uploader("Choose a file", type=None)

    if st.button("Ingest File", key="btn_ingest_file"):
        if file is None:
            st.error("Please select a file.")
        else:
            try:
                with st.spinner("Uploading & ingesting..."):
                    files = {"file": (file.name, file.getvalue())}
                    r = requests.post(f"{API_BASE}/ingest", files=files, timeout=300)
                if r.status_code == 200:
                    out = r.json()
                    st.success("Ingested successfully.")
                    st.json(out)
                    if auto_indexing is False:
                        st.warning("Auto indexing is OFF — go to Build Index tab to make it queryable.")
                    else:
                        st.info("If auto indexing is ON, this document should be queryable immediately.")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Ingest URL")
    url = st.text_input("Document URL")

    if st.button("Ingest URL", key="btn_ingest_url"):
        if not url.strip():
            st.error("Please enter a URL.")
        else:
            try:
                with st.spinner("Downloading & ingesting..."):
                    r = requests.post(f"{API_BASE}/ingest_url", json={"url": url.strip()}, timeout=300)
                if r.status_code == 200:
                    out = r.json()
                    st.success("Ingested successfully.")
                    st.json(out)
                    if auto_indexing is False:
                        st.warning("Auto indexing is OFF — go to Build Index tab to make it queryable.")
                    else:
                        st.info("If auto indexing is ON, this document should be queryable immediately.")
                else:
                    st.error(r.text)
                    st.info(
                        "If the site blocks downloads (403) or file type can't be inferred, "
                        "download locally and use Upload File."
                    )
            except Exception as e:
                st.error(str(e))

with tabs[2]:
    st.subheader("Build / Rebuild Index")

    if auto_indexing is True:
        st.info("Auto indexing is ON. You usually do NOT need this tab.")
        st.caption("Use it only if you want to force a rebuild (e.g., after many ingests).")
    elif auto_indexing is False:
        st.warning("Auto indexing is OFF. Run this after ingesting documents.")
    else:
        st.caption("If auto indexing is OFF, this is required after ingestion.")

    if st.button("Build Index", key="btn_build_index"):
        try:
            with st.spinner("Building FAISS index..."):
                r = requests.post(f"{API_BASE}/build_index", timeout=600)
            if r.status_code == 200:
                st.success("Index built successfully.")
                st.json(r.json())
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

with tabs[3]:
    st.subheader("Ingested Documents")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("Refresh", key="btn_docs_refresh"):
            st.rerun()
    with col_b:
        st.caption("Delete a doc here. If auto indexing is OFF, rebuild in the Build Index tab.")

    try:
        docs = fetch_documents()
    except Exception as e:
        docs = []
        st.error(f"Failed to load docs: {e}")

    if not docs:
        st.info("No docs found yet. Upload or ingest a URL first.")
    else:
        rows = []
        for d in docs:
            fn = d.get("original_filename") or "unknown"
            rows.append(
                {
                    "": icon_for(fn),
                    "Filename": fn,
                    "Chunks": d.get("chunks_count", 0),
                    "doc_id": d.get("doc_id", ""),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df[["", "Filename", "Chunks", "doc_id"]], width="stretch", hide_index=True)

        st.divider()
        st.subheader("Delete Document")

        options = [
            (d.get("doc_id", ""), d.get("original_filename") or d.get("doc_id", "unknown"))
            for d in docs
            if d.get("doc_id")
        ]
        label_to_id = {f"{icon_for(name)} {name}": doc_id for (doc_id, name) in options}
        selected_label = st.selectbox("Select a document", ["(choose one)"] + list(label_to_id.keys()))

        if st.button("Delete Selected", key="btn_delete_selected"):
            if selected_label == "(choose one)":
                st.error("Please select a document first.")
            else:
                doc_id_to_delete = label_to_id[selected_label]
                try:
                    with st.spinner("Deleting..."):
                        r = requests.delete(f"{API_BASE}/documents/{doc_id_to_delete}", timeout=120)
                    if r.status_code == 200:
                        st.success("Deleted successfully.")
                        st.json(r.json())
                        if auto_indexing is False:
                            st.warning("Auto indexing is OFF — rebuild index now if you plan to query.")
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

with tabs[4]:
    st.subheader("Query")

    try:
        docs = fetch_documents()
    except Exception:
        docs = []

    doc_label_map: dict[str, str] = {}
    doc_labels: List[str] = []
    for d in docs:
        doc_id = d.get("doc_id")
        fn = d.get("original_filename") or doc_id
        if doc_id:
            label = f"{icon_for(fn)} {fn} ({doc_id[:8]})"
            doc_label_map[label] = doc_id
            doc_labels.append(label)

    question = st.text_area("Question", height=120)

    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.number_input("top_k", min_value=1, max_value=8, value=5, step=1)
    with c2:
        default_min_score = float(min_score_default) if isinstance(min_score_default, (int, float)) else 0.25
        min_score = st.number_input("min_score", min_value=0.0, max_value=1.0, value=default_min_score, step=0.05)
    with c3:
        st.caption("Optional: filter retrieval to selected documents (multi-doc).")

    selected_docs = st.multiselect(
        "Document filter (optional)",
        options=doc_labels,
        default=[],
    )

    image = st.file_uploader("Optional image (OCR)", type=["png", "jpg", "jpeg"])

    if st.button("Ask", key="btn_ask"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            payload = {
                "question": question.strip(),
                "top_k": int(top_k),
                "min_score": float(min_score),
                "use_langchain": bool(use_langchain_ui),
                "return_context": bool(return_context_ui),
            }

            doc_ids = [doc_label_map[x] for x in selected_docs if x in doc_label_map]
            if doc_ids:
                payload["doc_ids"] = doc_ids

            if image is not None:
                payload["image_base64"] = base64.b64encode(image.getvalue()).decode("utf-8")

            try:
                with st.spinner("Searching + generating answer..."):
                    r = requests.post(f"{API_BASE}/query", json=payload, timeout=300)

                if r.status_code == 200:
                    out = r.json()

                    st.subheader("Answer")
                    st.write(out.get("answer", ""))

                    per_doc = out.get("per_document_stats", []) or []
                    if per_doc:
                        st.subheader("Per-document stats")
                        st.dataframe(pd.DataFrame(per_doc), width="stretch", hide_index=True)

                    if return_context_ui and out.get("context"):
                        st.subheader("Context")
                        st.code(out.get("context", ""), language="text")

                    st.subheader("Sources")
                    sources = out.get("sources", []) or []
                    if not sources:
                        st.info("No sources returned.")
                    else:
                        def group_key(s: dict) -> str:
                            fn = s.get("original_filename") or s.get("source") or "unknown"
                            did = s.get("doc_id")
                            if did:
                                return f"{icon_for(fn)} {fn} ({str(did)[:8]})"
                            return "🖼️ image_base64 (OCR)"

                        grouped: dict[str, list[dict]] = {}
                        for s in sources:
                            k = group_key(s)
                            grouped.setdefault(k, []).append(s)

                        for gname, items in grouped.items():
                            st.markdown(f"### {gname}")
                            for i, s in enumerate(items, start=1):
                                src = s.get("source", "unknown")
                                page = s.get("page", "?")
                                score = float(s.get("score", 0.0) or 0.0)
                                chunk_id = s.get("chunk_id", "?")
                                snippet = s.get("snippet", "") or ""
                                icon = s.get("icon", icon_for(src))
                                ftype = s.get("file_type", "unknown")

                                header = (
                                    f"**{i}. {icon} {src}**  \n"
                                    f"Type: **{ftype}** · Page: **{page}** · Chunk: **{chunk_id}** · Score: **{score:.3f}**"
                                )
                                with st.container(border=True):
                                    st.markdown(header)
                                    with st.expander("Snippet"):
                                        st.write(snippet)
                else:
                    st.error(r.text)
                    if "Index not built" in r.text and auto_indexing is False:
                        st.warning("Auto indexing is OFF — go to Build Index tab and build the index.")
            except Exception as e:
                st.error(str(e))
