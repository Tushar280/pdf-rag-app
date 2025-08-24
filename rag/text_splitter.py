from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_texts(pages_text):
    # Simple chunker, split per page, then by chunk size
    chunks = []
    for i, page in enumerate(pages_text):
        text = page or ""
        for j in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[j:j+CHUNK_SIZE]
            chunks.append({"id": f"{i}-{j}", "filename": "unknown", "page": i+1, "text": chunk})
    return chunks
