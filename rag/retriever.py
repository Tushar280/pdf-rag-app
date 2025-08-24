def retrieve(embedder, store, query, top_k=3):
    qvec = embedder.encode([query])
    return store.search(qvec, top_k)
