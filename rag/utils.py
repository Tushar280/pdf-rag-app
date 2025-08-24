import logging
def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)

logger = logging.getLogger("pdf-rag")
logging.basicConfig(level=logging.INFO)
