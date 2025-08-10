import streamlit as st
from typing import List, Dict, Optional
from services.ai_service import AIService
from services.advanced_retrieval import AdvancedRetrieval
from text_chunker.chunker import TextChunker
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SUPPORTED_LANGUAGES
from deep_translator import GoogleTranslator
import json
import os


class QAEngine:
    """Enhanced QA Engine with RAG + Feedback-Trained Prompts"""

    def __init__(self, model_name: str = "gemini", language: str = "english"):
        self.ai_service = AIService(model_name)
        self.language = language
        self.advanced_retrieval = AdvancedRetrieval()

        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="product_manual",
            embedding_function=self.embedding_function
        )

        # Load improved prompts from training step
        self.improved_prompts = self._load_improved_prompts()

        # RAG settings
        self.retrieval_strategy = "hybrid"
        self.prompt_style = "detailed"
        self.use_enhanced_context = True
        self.use_chain_of_thought = False

    def _load_improved_prompts(self) -> dict:
        """Load improved prompts from JSON if available."""
        path = "data/improved_prompts.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _get_custom_prompt(self, query: str) -> Optional[str]:
        """Return improved prompt if query matches a known issue keyword."""
        query_lower = query.lower()
        for issue, prompt in self.improved_prompts.items():
            if issue in query_lower:
                return prompt
        return None

    def process_document(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> bool:
        """Process and store chunks in vector DB."""
        try:
            self.collection.delete(where={})
            chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.chunk_text(text)
            documents, metadatas, ids = [], [], []

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"chunk_id": i, "chunk_size": len(chunk), "source": "product_manual"})
                ids.append(f"chunk_{i}")

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"✅ Processed {len(chunks)} chunks successfully")
            return True
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return False

    def ask(self, question: str, output_language: str = None) -> str:
        """Answer query using RAG + improved prompts if available."""
        try:
            language = output_language or self.language

            # Step 0: Check for improved prompt
            improved_prompt = self._get_custom_prompt(question)

            # Step 1: Retrieve relevant chunks
            chunks = self.get_similar_chunks(question, n_results=8)
            if self.use_enhanced_context:
                enhanced_result = self.advanced_retrieval.get_enhanced_context(
                    question, chunks, self.retrieval_strategy
                )
                context = enhanced_result['enhanced_context']
                retrieval_metadata = enhanced_result
            else:
                context = "\n\n".join(chunks)
                retrieval_metadata = {}

            # Step 2: Choose prompt
            if improved_prompt:
                print("⚡ Using improved prompt from feedback training")
                user_prompt = f"{improved_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}"
                system_prompt = "You are an improved assistant. Use the refined approach."
            else:
                system_prompt = "You are a helpful assistant. Provide accurate, detailed answers."
                user_prompt = (
                    f"Answer the following question in a {self.prompt_style} style.\n\n"
                    f"Context:\n{context}\n\nQuestion:\n{question}"
                )

            # Step 3: Generate response
            answer = self.ai_service.generate_response(user_prompt, system_prompt, language)

            # Step 4: Translate if needed
            if language.lower() != "english":
                answer = GoogleTranslator(source='auto', target=self.language_code(language)).translate(answer)

            # Step 5: Store metadata
            self._store_query_metadata(question, answer, retrieval_metadata, {"style": self.prompt_style})

            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

    def language_code(self, lang):
        return {"english": "en", "hindi": "hi", "marathi": "mr", "german": "de"}.get(lang.lower(), "en")

    def get_similar_chunks(self, query: str, n_results: int = 5) -> List[str]:
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            return []

    def _store_query_metadata(self, question: str, answer: str, retrieval_metadata: Dict, prompt_data: Dict):
        metadata = {
            'question': question,
            'answer': answer,
            'retrieval_strategy': retrieval_metadata.get('strategy', 'unknown'),
            'prompt_style': prompt_data.get('style', 'unknown'),
            'chunks_analyzed': retrieval_metadata.get('chunks_analyzed', 0),
            'top_chunks': retrieval_metadata.get('top_chunks', 0),
            'timestamp': datetime.now().isoformat()
        }
        if 'query_metadata' not in st.session_state:
            st.session_state.query_metadata = []
        st.session_state.query_metadata.append(metadata)
 