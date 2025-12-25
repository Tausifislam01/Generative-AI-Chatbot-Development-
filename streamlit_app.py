import os
import base64
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="RAG Assignment UI", layout="wide")
st.title("RAG Assignment UI")

tabs = st.tabs(["Upload File", "Ingest URL", "Build Index", "Docs", "Query"])

with tabs[0]:
    st.subheader("Upload File")
    file = st.file_uploader("Choose a file", type=None)
    if st.button("Ingest File"):
        if file is None:
            st.error("Please select a file.")
        else:
            try:
                files = {"file": (file.name, file.getvalue())}
                r = requests.post(f"{API_BASE}/ingest", files=files, timeout=300)
                if r.status_code == 200:
                    st.success("Ingested successfully.")
                    st.json(r.json())
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Ingest URL")
    url = st.text_input("Document URL")
    if st.button("Ingest URL"):
        if not url.strip():
            st.error("Please enter a URL.")
        else:
            try:
                r = requests.post(f"{API_BASE}/ingest_url", json={"url": url.strip()}, timeout=300)
                if r.status_code == 200:
                    st.success("Ingested successfully.")
                    st.json(r.json())
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

with tabs[2]:
    st.subheader("Build / Rebuild Index")
    if st.button("Build Index"):
        try:
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
    if st.button("Refresh Documents"):
        try:
            r = requests.get(f"{API_BASE}/documents", timeout=60)
            if r.status_code == 200:
                docs = r.json()
                if docs:
                    st.json(docs)
                else:
                    st.info("No docs found yet. Upload or ingest a URL first.")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"Failed to load docs: {e}")

    st.subheader("Delete Document")
    doc_id_to_delete = st.text_input("Document ID to delete")
    if st.button("Delete"):
        if not doc_id_to_delete.strip():
            st.error("Please enter a document ID.")
        else:
            try:
                r = requests.delete(f"{API_BASE}/documents/{doc_id_to_delete.strip()}", timeout=120)
                if r.status_code == 200:
                    st.success("Deleted successfully.")
                    st.json(r.json())
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

with tabs[4]:
    st.subheader("Query")
    question = st.text_area("Question", height=120)

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5, step=1)
    with col2:
        doc_id = st.text_input("doc_id (optional)")
    with col3:
        min_score = st.number_input("min_score", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    image = st.file_uploader("Optional image (OCR)", type=["png", "jpg", "jpeg"])

    if st.button("Ask"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            payload = {"question": question.strip(), "top_k": int(top_k), "min_score": float(min_score)}
            if doc_id.strip():
                payload["doc_id"] = doc_id.strip()
            if image is not None:
                payload["image_base64"] = base64.b64encode(image.getvalue()).decode("utf-8")

            try:
                r = requests.post(f"{API_BASE}/query", json=payload, timeout=300)
                if r.status_code == 200:
                    out = r.json()
                    st.subheader("Answer")
                    st.write(out.get("answer", ""))
                    st.subheader("Sources")
                    st.json(out.get("sources", []))
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))
