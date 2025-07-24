# text_chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the text chunker with configurable chunk size and overlap.
        :param chunk_size: Number of characters in each chunk
        :param chunk_overlap: Number of characters overlapping between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits the input text into smaller overlapping chunks.
        :param text: Complete text extracted from the PDF
        :return: List of chunked text strings
        """
        return self.splitter.split_text(text)
    
    def save_chunks(self, chunks: List[str], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"\n--- Chunk {i + 1} ---\n{chunk}\n")

    def __repr__(self):
        return f"TextChunker(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"



# Example usage
if __name__ == "__main__":

    sample_text = "This is a long manual or document that needs to be split into smaller parts for processing..."
    
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_text(sample_text)

    print(f"Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
        print(f"\n--- Chunk {i + 1} ---\n{chunk}")