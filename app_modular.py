"""
Modular Product Manual Assistant Application
"""
import streamlit as st
from core.qa_engine import QAEngine
from ui.components import UIComponents
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class ProductManualAssistant:
    """Main application class for Product Manual Assistant"""
    
    def __init__(self):
        self.ui = UIComponents()
        #created for each session to store vector DB
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        # Initialize UI state variables
        if "input_language" not in st.session_state:
            st.session_state.input_language = "english"
        if "output_language" not in st.session_state:
            st.session_state.output_language = "english"
        if "ai_model" not in st.session_state:
            st.session_state.ai_model = "ollama"
        if "qa_engine_initialized" not in st.session_state:
            st.session_state.qa_engine_initialized = False
    
    def run(self):
        """Main application runner"""
        # Render header
        self.ui.render_header()
        
        # Define tabs
        tab1, tab2, tab3 = st.tabs(["Upload & Process", "Ask Questions", "Feedback & Team"])
        
        # Tab 1: Upload & Process
        with tab1:
            self._handle_pdf_upload()
        
        # Tab 2: Ask Questions
        with tab2:
            self._handle_qa_section()
        
        # Tab 3: Feedback & Team
        with tab3:
            self.ui.render_feedback_section()
    
    def _handle_pdf_upload(self):
        """Handle PDF upload and processing"""
        extracted_text = self.ui.render_pdf_upload()
        
        if extracted_text:
            # Process text
            with st.spinner("Processing text..."):
                # Initialize QA engine
                qa_engine = QAEngine(model_name=st.session_state.ai_model, language=st.session_state.input_language)
                qa_engine.process_document(extracted_text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
                
                # Save to session state
                st.session_state.extracted_text = extracted_text
                st.session_state.qa_engine = qa_engine
                st.session_state.qa_engine_initialized = True
                
                st.success("✅ Document processed and stored in vector database successfully!")
    
    def _handle_qa_section(self):
        """Handle question-answering section"""
        if "qa_engine" not in st.session_state or not st.session_state.qa_engine_initialized:
            st.warning("Please upload and process a PDF in the first tab.")
            return
        
        # Language and model selection with state management
        input_language, output_language = self.ui.render_language_selection()
        ai_model = self.ui.render_model_selection()
        
        # Only update QA engine if model or language actually changed
        needs_update = (
            st.session_state.input_language != input_language or
            st.session_state.ai_model != ai_model
        )
        
        if needs_update:
            with st.spinner("Updating AI model configuration..."):
                st.session_state.input_language = input_language
                st.session_state.output_language = output_language
                st.session_state.ai_model = ai_model
                
                # Update the existing QA engine instead of recreating it
                st.session_state.qa_engine.language = input_language
                
                # Update AI service model
                st.session_state.qa_engine.ai_service.model_name = ai_model
                st.session_state.qa_engine.ai_service._initialize_model()
                
                st.success(f"✅ Switched to {ai_model.upper()} model with {input_language.upper()} language!")
        
        # Render QA section
        self.ui.render_qa_section(st.session_state.qa_engine, ai_model, output_language)


def main():
    """Main entry point"""
    app = ProductManualAssistant()
    app.run()


if __name__ == "__main__":
    main() 