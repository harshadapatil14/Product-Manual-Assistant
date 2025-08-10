"""
UI Components for Product Manual Assistant
"""
import streamlit as st
import uuid
import os
import plotly.express as px
from typing import Optional
from services.pdf_service import PDFService
from services.sentiment_service import SentimentService
from services.feedback_service import FeedbackService
from core.qa_engine import QAEngine
from config.settings import SUPPORTED_LANGUAGES, AI_MODELS, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
import pandas as pd
import speech_recognition as sr
from typing import Optional

class UIComponents:
    """UI Components for the Streamlit interface"""
    
    def __init__(self):
        self.pdf_service = PDFService()
        self.sentiment_service = SentimentService()
        self.feedback_service = FeedbackService()
        # self.model_training_service = ModelTrainingService()
        # Removed: self.fine_tuning_service = FineTuningService()
    
    def render_header(self):
        """Render the main header"""
        st.set_page_config(page_title="Product Manual Assistant", layout="wide")
        st.title("Product Manual Assistant")
        st.markdown("📖 **Text-based** | 🤖 **AI-Powered** | 🌐 **Multi-Language**")
    
    def render_language_selection(self) -> tuple:
        """Render language selection components"""
        st.subheader("🌐 Language Settings")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            input_language = st.selectbox(
                "Input Language:",
                list(SUPPORTED_LANGUAGES.keys()),
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.get("input_language", "english")),
                format_func=lambda x: SUPPORTED_LANGUAGES[x]['name'],
                help="Select the language for your question"
            )
        
        with col2:
            output_language = st.selectbox(
                "Output Language:",
                list(SUPPORTED_LANGUAGES.keys()),
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.get("output_language", "english")),
                format_func=lambda x: SUPPORTED_LANGUAGES[x]['name'],
                help="Select the language for the answer"
            )
        
        return input_language, output_language
    
    def render_model_selection(self) -> str:
        """Render AI model selection"""
        st.subheader("🤖 AI Model Selection")
        ai_model = st.selectbox(
            "Choose your AI model:",
            list(AI_MODELS.keys()),
            index=list(AI_MODELS.keys()).index(st.session_state.get("ai_model", "gemini")),
            format_func=lambda x: AI_MODELS[x]['name'],
            help="Select which AI model to use for generating answers"
        )
        return ai_model
    
    def render_voice_input(self) -> Optional[str]:
        """Render voice input section - ENABLED & auto-fill text box"""

        st.subheader("🎤 Voice Input")

        # Create a button to start recording
        if st.button("Start Recording"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... Please speak now.")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source)

            try:
                st.info("Transcribing your speech...")
                text = recognizer.recognize_google(audio)
                st.success("Transcription complete!")
                st.write(f"**You said:** {text}")

                # Auto-fill the text box in render_text_input
                st.session_state.current_query = text

                return text
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand the audio.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")

        return ""
    
    def render_text_input(self, default_value: str = '') -> str:
        """Render text input section"""
        st.subheader("✍️ Text Input")
        
        # Use session state to maintain query across re-renders
        if "current_query" not in st.session_state:
            st.session_state.current_query = default_value
        
        query = st.text_input("Type your question:", value=st.session_state.current_query, key="query_input")
        
        # Update session state
        st.session_state.current_query = query
        
        return query
    
    def render_pdf_upload(self):
        """Render PDF upload section"""
        st.header("Upload a Product Manual PDF")
        uploaded_file = st.file_uploader("Upload PDF Manual", type=["pdf"])
        
        if uploaded_file:
            # Validate PDF
            if not self.pdf_service.validate_pdf(uploaded_file):
                st.error("❌ Invalid PDF file. Please upload a valid PDF.")
                return None
            
            # Extract text
            with st.spinner("Extracting text from PDF..."):
                text = self.pdf_service.extract_text_from_pdf(uploaded_file)
            
            if text:
                st.success("✅ Text extracted successfully.")
                
                # Get PDF info
                pdf_info = self.pdf_service.get_pdf_info(uploaded_file)
                st.info(f"📄 PDF Info: {pdf_info.get('num_pages', 0)} pages, {pdf_info.get('file_size', 0)} bytes")
                
                return text
            else:
                st.error("❌ Failed to extract text from PDF.")
                return None
        
        return None
    
    def render_qa_section(self, qa_engine: QAEngine, ai_model: str, output_language: str):
        """Render question-answering section"""
        st.header("Ask a Question")
        
        # Voice input
        voice_input = self.render_voice_input()
        
        # Text input
        query = self.render_text_input(voice_input)
        
        if st.button("Get Answer"):
            if query:
                with st.spinner(f"Generating answer using {ai_model.upper()} model in {output_language.upper()}..."):
                    try:
                        answer = qa_engine.ask(query, output_language=output_language)
                    
                        # Display answer
                        st.markdown("### Answer:")
                        st.write(answer)
                        
                        # Show model and language info
                        st.info(f"🤖 Answer generated using: **{ai_model.upper()}** in **{output_language.upper()}**")
                        

                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
                        st.exception(e)
                
                # Show context if requested
                if st.checkbox("Show retrieved context"):
                    try:
                        similar_chunks = qa_engine.get_similar_chunks(query, n_results=3)
                        for i, chunk in enumerate(similar_chunks):
                            st.markdown(f"**Chunk {i + 1}:**")
                            st.code(chunk[:1000])
                    except Exception as e:
                        st.error(f"❌ Error retrieving context: {str(e)}")

    def render_feedback_section(self):
        """Render feedback section with ratings chart and sentiment breakdown"""
        st.markdown("We value your feedback! Please share your thoughts to help us improve.")

        # Team info
        st.markdown("### 👥 Team")
        st.markdown("**Harshada Patil** — 📧 harshadaavijaypatil@gmail.com")
        st.markdown("**Pallavi Dudhalkar** — 📧 pallavi.dudhalkar@gmail.com")

        # Load existing feedback
        feedback_data = self.feedback_service.load_all_feedback()
        ratings = [fb['rating'] for fb in feedback_data if 'rating' in fb]
        sentiments = [fb.get('sentiment', 'Neutral') for fb in feedback_data]

    


       
        # Show ratings overview
        if ratings:
            st.subheader("⭐ Ratings Overview")
            rating_df = pd.DataFrame(ratings, columns=["Rating"])


            rating_counts = rating_df["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            df = rating_counts.reset_index()
            df.columns = ['Rating', 'Count']

            fig = px.bar(df, x='Count', y='Rating', orientation='h',
                        category_orders={'Rating': [1, 2, 3, 4, 5]})

            st.plotly_chart(fig, use_container_width=True)

            # Ensure all star counts 1–5 appear (even if 0)
            # rating_counts = rating_df["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            # st.bar_chart(rating_counts)
        else:
            st.info("No ratings yet. Be the first to give feedback!")

        st.markdown("---")
        st.subheader("✍️ Share Your Feedback")

        # Rating slider
        rating = st.slider("How would you rate the application?", 1, 5, 3)

        # Comment box
        comment = st.text_area("Additional Comments (optional):", placeholder="Tell us what could be improved...")

        if st.button("Submit Feedback"):
            feedback_data_entry = {
                "rating": rating,
                "comment": comment.strip(),
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "type": "general_feedback"  # Helps filter later
            }

            # Optional: Analyze sentiment if comment provided
            if comment.strip():
                sentiment_result = self.sentiment_service.analyze_sentiment(comment)
                feedback_data_entry['sentiment'] = sentiment_result['final_sentiment']['sentiment']

            if self.feedback_service.save_feedback(feedback_data_entry):
                st.success("✅ Thank you for your feedback!")
            else:
                st.error("❌ Failed to save feedback")
