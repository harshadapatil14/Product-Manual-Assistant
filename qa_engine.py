# qa_engine.py

from typing import List
import requests
from text_chunker.embedding_generator import EmbeddingGenerator
from text_chunker.vector_store import VectorStore

class QAEngine:
    def __init__(self, persist_directory: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = EmbeddingGenerator(model_name)
        self.vector_store = VectorStore(persist_directory)

    def ask(self, query: str, top_k: int = 10) -> str:
        print(f"[+] Query received: {query}")

        # Step 1: Retrieve relevant chunks
        similar_chunks = self.vector_store.query(query, n_results=top_k)

        if not similar_chunks:
            return "Sorry, I couldn't find relevant information in the document."

        # Step 2: Construct prompt for local LLM (Ollama)
        context = "\n\n".join(similar_chunks)
        prompt = f"""
You are a helpful assistant for a product manual. Use the context below to answer the question with accurate, complete detail.

- Break your answer into **clear step-b
y-step instructions**.
- If a step involves safety or caution, mention it.
- If the information is unavailable, say "Not found in the document."

### Document:
{context}

### Question:
{query}

### Answer (Step-by-step):
1.
"""


        # Step 3: Call Ollama model (e.g., llama3)
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from model.")
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"An error occurred while generating the answer: {str(e)}"
