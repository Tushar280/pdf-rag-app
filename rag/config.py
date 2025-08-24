import os

OPENAI_API_KEY = os.getenv("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", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

VECTORSTORE_BACKEND = os.getenv("VECTORSTORE_BACKEND", "chroma")
FAISS_INDEX_DIR = "data/index"
CHROMA_PERSIST_DIR = "data/chroma"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", 2000))
TOP_K = int(os.getenv("TOP_K", 3))
