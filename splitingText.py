from pdf_processor import PDFPlumberProcessor
from text_chunker import TextChunker

# Step 1: Extract text from PDF
processor = PDFPlumberProcessor("R6400_UM.pdf")
full_text, raw_text = processor.process_pdf()

# Step 2: Chunk the extracted full text
chunker = TextChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_text(full_text)

# (Optional) Print or save the chunks
print(f"Total Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---\n{chunk}")
