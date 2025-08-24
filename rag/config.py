import os

LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # default changed to "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# The below local LLM config can remain but won't be used if backend is openai
LOCAL_LLM_MODEL_PATH = os.getenv("LOCAL_LLM_MODEL_PATH", "models/Mistral-7B-Instruct.Q4_K_M.gguf")
LOCAL_LLM_MODEL_TYPE = os.getenv("LOCAL_LLM_MODEL_TYPE", "mistral")
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", 384))
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", 0.2))

VECTORSTORE_BACKEND = os.getenv("VECTORSTORE_BACKEND", "chroma")

FAISS_INDEX_DIR = "data/index"
CHROMA_PERSIST_DIR = "data/chroma"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", 2000))
TOP_K = int(os.getenv("TOP_K", 3))
