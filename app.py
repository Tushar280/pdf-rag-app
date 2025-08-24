import os
import streamlit as st
from rag.rag_pipeline import ingest_pdfs, answer_question
from rag.config import PDF_STORAGE_DIR
from rag.utils import ensure_dir

st.set_page_config(page_title="PDF RAG App", page_icon="📄", layout="wide")

st.title("📄 PDF RAG App")
st.caption("Upload PDFs → Embed → Ask questions with sources and page numbers")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Select one or more PDFs", accept_multiple_files=True, type=["pdf"])
    if st.button("Ingest PDFs"):
        if not uploaded:
            st.warning("Please select at least one PDF.")
        else:
            ensure_dir(PDF_STORAGE_DIR)
            local_paths = []
            for file in uploaded:
                path = os.path.join(PDF_STORAGE_DIR, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                local_paths.append(path)
            with st.status("Ingesting PDFs...", expanded=True):
                res = ingest_pdfs(local_paths)
            st.success(f"Ingested {res['chunks_added']} chunks.")

st.header("Ask a question")
query = st.text_input("Enter your question about the uploaded PDFs")

col1, col2 = st.columns([1,2])
with col1:
    top_k = st.slider("Top-K", 1, 10, 5)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            res = answer_question(query, top_k=top_k)
        st.subheader("Answer")
        st.write(res["answer"])

        st.subheader("Sources")
        for s in res["sources"]:
            st.markdown(f"- {s['filename']} — page {s['page']} — score {s['score']:.3f}")
            st.code(s["snippet"])
