from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import faiss


class FaissVectorStore:

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "meta.json"

        self.index: faiss.Index | None = None
        self.meta: List[Dict[str, Any]] = []

    def build(self, vectors: np.ndarray, meta: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D: shape [n, d]")
        if len(meta) != vectors.shape[0]:
            raise ValueError("meta length must match number of vectors")

        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(vectors)

        self.index = index
        self.meta = meta

        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.index_path.exists() or not self.meta_path.exists():
            return False

        self.index = faiss.read_index(str(self.index_path))
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return True

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call /build_index first.")
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("query_vec must be shape [1, d]")

        scores, ids = self.index.search(query_vec, top_k)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((float(score), self.meta[idx]))
        return results
