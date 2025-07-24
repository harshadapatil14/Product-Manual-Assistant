# vector_store.py

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_data"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="manual_chunks",
            embedding_function=self.embedding_func
        )

    def add_documents(self, texts: List[str]):
        """
        Stores chunks and their embeddings into ChromaDB.
        """
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(documents=texts, ids=ids)

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """
        Searches for top-N most relevant chunks using semantic similarity.
        Returns only the document text.
        """
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        return results["documents"][0]  # This is a list of matching chunks

