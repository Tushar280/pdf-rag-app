from sentence_transformers import SentenceTransformer
import numpy as np

SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformerEmbeddings:
    def __init__(self, model_name=SENTENCE_TRANSFORMERS_MODEL):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
