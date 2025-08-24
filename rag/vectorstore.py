import os
import numpy as np
from typing import List, Dict, Tuple
from rag.config import VECTORSTORE_BACKEND, FAISS_INDEX_DIR, CHROMA_PERSIST_DIR
from rag.utils import ensure_dir, logger

class VectorStoreBase:
    def add(self, embeddings: np.ndarray, metadatas: List[Dict], ids: List[str]):
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        raise NotImplementedError

    def persist(self):
        raise NotImplementedError

class ChromaStore(VectorStoreBase):
    def __init__(self, dim: int, persist_dir: str = CHROMA_PERSIST_DIR):
        import chromadb
        from chromadb.config import Settings

        ensure_dir(persist_dir)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        self.collection = self.client.get_or_create_collection("pdf_chunks")
        self.dim = dim

    def add(self, embeddings: np.ndarray, metadatas: List[Dict], ids: List[str]):
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        res = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        results: List[Tuple[Dict, float]] = []
        if not res:
            return results

        metadatas_batches = res.get("metadatas") or []
        distances_batches = res.get("distances") or []

        if not metadatas_batches or not distances_batches:
            return results

        metadatas = metadatas_batches[0] or []
        distances = distances_batches[0] or []

        def to_float_distance(d):
            if d is None:
                return None
            if isinstance(d, (int, float)):
                return float(d)
            if isinstance(d, (list, tuple)) and d:
                first = d[0]
                return float(first) if isinstance(first, (int, float)) else None
            return None

        for md, dist in zip(metadatas, distances):
            d = to_float_distance(dist)
            score = float(1.0 - d) if d is not None else 0.0
            results.append((md, score))
        return results

    def persist(self):
        pass  # Persistence is automatic for Chroma

class FaissStore(VectorStoreBase):
    def __init__(self, dim: int, index_dir: str = FAISS_INDEX_DIR):
        import faiss
        ensure_dir(index_dir)
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadata.npy")
        self.ids_path = os.path.join(index_dir, "ids.npy")
        self.faiss = faiss
        self.dim = dim

        if os.path.exists(self.index_path):
            logger.info("Loading FAISS index from disk")
            self.index = faiss.read_index(self.index_path)
            self.metadata = np.load(self.meta_path, allow_pickle=True).tolist()
            self.ids = np.load(self.ids_path, allow_pickle=True).tolist()
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatIP(dim)
            self.metadata: List[Dict] = []
            self.ids: List[str] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict], ids: List[str]):
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        D, I = self.index.search(query_vec.astype(np.float32), top_k)
        results: List[Tuple[Dict, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(dist)))
        return results

    def persist(self):
        self.faiss.write_index(self.index, self.index_path)
        np.save(self.meta_path, np.array(self.metadata, dtype=object))
        np.save(self.ids_path, np.array(self.ids, dtype=object))

def get_vectorstore(dim: int) -> VectorStoreBase:
    if VECTORSTORE_BACKEND == "chroma":
        return ChromaStore(dim)
    return FaissStore(dim)
