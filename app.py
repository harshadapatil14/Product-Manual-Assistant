import streamlit as st
import os
import uuid
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pdfplumber import open as pdf_open

from text_chunker.chunker import TextChunker
from text_chunker.vector_store import VectorStore
from qa_engine import QAEngine
from pdf_processor import PDFPlumberProcessor

# -----------------------------
# Function: Extract text from PDF using pdfplumber
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    all_text = ""
    with pdf_open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            all_text += f"\n\n--- Page {i + 1} ---\n{page_text or ''}"
    return all_text

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Product Manual Assistant", layout="wide")
st.title("📘 Product Manual Assistant")

# -----------------------------
# Define Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "❓ Ask Questions", "📣 Feedback & Team"])

# -----------------------------
# TAB 1: Upload & Process PDF
# -----------------------------
with tab1:
    st.header("Step 1: Upload a Product Manual PDF")
    uploaded_file = st.file_uploader("Upload PDF Manual", type=["pdf"])

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        st.success("✅ Text extracted successfully.")

        # Create persist directory using session state
        if "persist_dir" not in st.session_state:
            session_id = str(uuid.uuid4())
            st.session_state.persist_dir = f"session_data/{session_id}"
            os.makedirs(st.session_state.persist_dir, exist_ok=True)

        # Chunk and store text
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_text(text)

        store = VectorStore(persist_directory=st.session_state.persist_dir)
        store.add_documents(chunks)

        # Save extracted text for later use
        st.session_state.extracted_text = text
        st.session_state.qa_engine = QAEngine(persist_directory=st.session_state.persist_dir)

# -----------------------------
# TAB 2: Ask Questions
# -----------------------------
with tab2:
    st.header("Step 2: Ask a Question")
    if "qa_engine" not in st.session_state:
        st.warning("Please upload and process a PDF in the first tab.")
    else:
        query = st.text_input("Type your question:")
        if st.button("Get Answer"):
            with st.spinner("Generating answer using local model..."):
                answer = st.session_state.qa_engine.ask(query)
                st.markdown("### Answer:")
                st.write(answer)

                # Show context if requested
                if st.checkbox("Show retrieved context"):
                    similar_chunks = st.session_state.qa_engine.vector_store.query(query, n_results=3)
                    for i, chunk in enumerate(similar_chunks):
                        st.markdown(f"**Chunk {i + 1}:**")
                        st.code(chunk[:1000])

# -----------------------------
# TAB 3: Feedback, Visualization & Team Info
# -----------------------------
with tab3:
    st.header("Step 3: Feedback on the Answer")

    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    sentiment_analyzer = SentimentIntensityAnalyzer()

    feedback_text = st.text_area("Was this answer helpful? Share your feedback below:")

    if st.button("Submit Feedback"):
        if feedback_text.strip() == "":
            st.warning("Please enter feedback before submitting.")
        else:
            sentiment = sentiment_analyzer.polarity_scores(feedback_text)
            compound = sentiment['compound']

            if compound > 0.05:
                result = "✅ Positive feedback detected: Model rewarded."
            elif compound < -0.05:
                result = "❌ Negative feedback detected: Model penalized."
            else:
                result = "😐 Neutral feedback received."

            st.success(result)

            # Save to log
            os.makedirs("logs", exist_ok=True)
            with open("logs/feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Feedback: {feedback_text}\nSentiment: {compound}\nResult: {result}\n\n")

    st.markdown("---")
    st.subheader("📊 Feedback Visualization")

    try:
        with open("logs/feedback_log.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        feedbacks = []
        for i in range(0, len(lines), 4):
            feedback = lines[i].replace("Feedback: ", "").strip()
            sentiment_score = float(lines[i + 1].split(":")[1].strip())
            result = lines[i + 2].split(":")[1].strip()
            feedbacks.append({
                "Feedback": feedback,
                "Sentiment Score": sentiment_score,
                "Result": result
            })

        df = pd.DataFrame(feedbacks)

        st.subheader("Feedback Categories")
        fig1, ax1 = plt.subplots()
        df['Result'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, ylabel='', colors=['#8fd9a8', '#f49f9f', '#fbe7a1'])
        st.pyplot(fig1)

        st.subheader("Sentiment Score Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["Sentiment Score"], bins=10, kde=True, ax=ax2, color="skyblue")
        st.pyplot(fig2)

        if st.checkbox("Show Raw Feedback Log"):
            st.dataframe(df)

    except FileNotFoundError:
        st.info("No feedback available yet.")

    st.markdown("---")
    st.subheader("Team")
    st.markdown("""
    **Harshada Patil**  
    📧 harshadaavijaypatil@example.com

    **Pallavi Dudhalkar**  
    📧 pallavi.dudhalkar@example.com
    """)
