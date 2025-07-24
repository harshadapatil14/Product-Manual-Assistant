# main.py

from text_chunker.chunker import TextChunker
from text_chunker.embedding_generator import EmbeddingGenerator
from text_chunker.vector_store import VectorStore
from qa_engine import QAEngine

# Step 1: Load pre-extracted text
with open("full_text_plumber.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Step 2: Chunk the text
chunker = TextChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_text(full_text)

# Step 3: Embed the chunks
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings(chunks)

# Step 4: Store in ChromaDB
store = VectorStore(persist_directory="data")
store.add_documents(chunks)

# Step 5: Initialize QA Engine and Ask Question
qa = QAEngine(persist_directory="data")
question = "How do I install the NETGEAR R6400 router for the first time"
answer = qa.ask(question)

# Step 6: Show Answer
print("\nAnswer:\n", answer)

