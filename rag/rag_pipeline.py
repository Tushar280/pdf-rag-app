from rag.embeddings import SentenceTransformerEmbeddings
from rag.vectorstore import get_vectorstore
from rag.pdf_loader import extract_text_by_page
from rag.text_splitter import chunk_texts
from rag.retriever import retrieve
from rag.llm import get_llm, make_prompt

def ingest_pdfs(paths):
    embedder = SentenceTransformerEmbeddings()
    all_chunks = []
    for path in paths:
        texts = extract_text_by_page(path)
        chunks = chunk_texts(texts)
        all_chunks.extend(chunks)
    store = get_vectorstore(384)
    embeddings = embedder.encode([c['text'] for c in all_chunks])
    ids = [c['id'] for c in all_chunks]
    metadatas = [{"filename": c['filename'], "page": c['page'], "text": c['text']} for c in all_chunks]
    store.add(embeddings, metadatas, ids)
    store.persist()
    return {"chunks_added": all_chunks}

def answer_question(query, top_k=3):
    embedder = SentenceTransformerEmbeddings()
    store = get_vectorstore(384)
    docs = retrieve(embedder, store, query, top_k=top_k)
    prompt = make_prompt(query, docs)
    llm = get_llm()
    answer = llm.generate(prompt)
    sources = [f"{d['filename']}:{d['page']}" for d, _ in docs]
    return {"answer": answer, "sources": sources}
