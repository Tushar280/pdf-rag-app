import streamlit as st
from rag.rag_pipeline import ingest_pdfs, answer_question
import os

st.title("PDF RAG App – Ask Questions From Your PDFs")

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type='pdf', accept_multiple_files=True)
if uploaded_files:
    local_paths = []
    for file in uploaded_files:
        path = f"data/pdfs/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())
        local_paths.append(path)
    if st.button("Ingest PDFs"):
        with st.spinner("Ingesting PDFs..."):
            res = ingest_pdfs(local_paths)
            st.success(f"Ingested {len(res['chunks_added'])} chunks.")

query = st.text_input("Ask a question:")
top_k = st.slider("Retrieved Chunks", 1, 10, 3)
if st.button("Get Answer"):
    with st.spinner("RAG pipeline running..."):
        res = answer_question(query, top_k=top_k)
        st.markdown("### Answer")
        st.write(res["answer"])
        st.markdown("### Sources")
        st.write("\n".join(res["sources"]))
