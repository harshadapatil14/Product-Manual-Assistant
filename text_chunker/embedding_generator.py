# embedding_generator.py

from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of text chunks into vector embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()
