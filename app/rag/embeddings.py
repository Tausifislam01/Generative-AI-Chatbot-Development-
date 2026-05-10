import os
from typing import List
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Embedder:

    def __init__(self, model_name: str = "gemini-embedding-2"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        self.model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            vecs.append(self.model.embed_query(text))
        return np.array(vecs, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.embed_query(query)
        return np.array([vec], dtype="float32")
