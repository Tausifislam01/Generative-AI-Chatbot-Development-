from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")
