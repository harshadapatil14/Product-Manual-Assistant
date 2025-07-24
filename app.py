import streamlit as st
import os
import uuid
from pathlib import Path
import shutil

from text_chunker.chunker import TextChunker
from text_chunker.vector_store import VectorStore
from qa_engine import QAEngine
from pdfplumber import open as pdf_open


# -----------------------------
# Function: Extract text from uploaded PDF using pdfplumber
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    all_text = ""
    with pdf_open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            all_text += f"\n\n--- Page {i + 1} ---\n{page_text or ''}"
    return all_text


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Manual Assistant", layout="wide")
st.title("Product Manual Assistant")
st.write("Upload a product manual (PDF) and ask questions about it.")


# -----------------------------
# Step 1: Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF Manual", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    # -----------------------------
    # Step 2: Create unique directory for vector store
    # -----------------------------
    persist_dir = f"session_data/{uuid.uuid4()}"
    os.makedirs(persist_dir, exist_ok=True)

    st.success("✅ Text extracted successfully.")

    # -----------------------------
    # Step 3: Chunk the extracted text
    # -----------------------------
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_text(text)

    # -----------------------------
    # Step 4: Store chunks in vector DB
    # -----------------------------
    # Use a consistent persist directory per Streamlit session
    if "persist_dir" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state.persist_dir = f"session_data/{session_id}"
        os.makedirs(st.session_state.persist_dir, exist_ok=True)

    persist_dir = st.session_state.persist_dir


    # if os.path.exists(persist_dir):
    #     shutil.rmtree(persist_dir)
    # os.makedirs(persist_dir, exist_ok=True)

    store = VectorStore(persist_directory=persist_dir)
    store.add_documents(chunks)

    # -----------------------------
    # Step 5: Initialize QA engine
    # -----------------------------
    qa = QAEngine(persist_directory=persist_dir)

    # -----------------------------
    # Step 6: Accept user query
    # -----------------------------
    st.subheader("Ask a Question")
    query = st.text_input("Type your question about the manual:")

    if st.button("Get Answer") and query:
        with st.spinner("Generating answer using local model..."):
            answer = qa.ask(query)
            st.markdown("### Answer:")
            st.write(answer)

    # feedback feature 
    # 1. Required imports
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import streamlit as st
    import os

    # 2. Download VADER lexicon (only once)
    nltk.download('vader_lexicon')

    # 3. Initialize sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # 4. Feedback section in your Streamlit app (put this after displaying the model's answer)
    st.subheader("📣 Feedback on Answer")

    feedback_text = st.text_area("Was this answer helpful? Share your feedback below:")

    if st.button("Submit Feedback"):
        if feedback_text.strip() == "":
            st.warning("Please enter feedback before submitting.")
        else:
            sentiment = sentiment_analyzer.polarity_scores(feedback_text)
            compound = sentiment['compound']

            if compound > 0.05:
                result = "✅ Positive feedback detected: Model rewarded."
                # Optional: Increment a reward counter or score
            elif compound < -0.05:
                result = "❌ Negative feedback detected: Model penalized."
                # Optional: Log penalty, degrade score, or flag
            else:
                result = "😐 Neutral feedback received."

            st.success(result)

            # Optional: Save feedback to log file
            os.makedirs("logs", exist_ok=True)
            with open("logs/feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Feedback: {feedback_text}\nSentiment: {compound}\nResult: {result}\n\n")

