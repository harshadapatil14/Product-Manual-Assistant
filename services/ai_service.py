"""
AI service for handling different AI model interactions
"""
import requests
import google.generativeai as genai
from typing import Optional, Dict, Any
from config.settings import GEMINI_API_KEY, AI_MODELS, SUPPORTED_LANGUAGES
import streamlit as st


class AIService:
    """Handles interactions with different AI models"""
    
    def __init__(self, model_name: str = "gemini"):
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected AI model"""
        if self.model_name == "gemini":
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(AI_MODELS['gemini']['model_name'])
            else:
                print("GEMINI_API_KEY not found. Falling back to Ollama.")
                self.model_name = "ollama"
    
    def generate_response(self, query: str, context: str, language: str = "english") -> str:
        """
        Generate response using the selected AI model
        
        Args:
            query: User's question
            context: Retrieved document chunks
            language: Language for response
            
        Returns:
            Generated response text
        """
        if self.model_name == "gemini":
            return self._generate_gemini_response(query, context, language)
        else:
            return self._generate_ollama_response(query, context, language)
    
    def _generate_gemini_response(self, query: str, context: str, language: str) -> str:
        """Generate response using Gemini API"""
        try:
            lang_config = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["english"])
            
            prompt = f"""
{lang_config['system']}

{lang_config['instructions']}

Context:
{context}

Question: {query}

Answer (Step-by-step):
"""
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"An error occurred while generating the answer with Gemini: {str(e)}"
    
    def _generate_ollama_response(self, query: str, context: str, language: str) -> str:
        """Generate response using Ollama (local LLM)"""
        try:
            lang_config = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["english"])
            
            prompt = f"""
{lang_config['system']}

{lang_config['instructions']}

### Document:
{context}

### Question:
{query}

### Answer (Step-by-step):
1.
"""
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": AI_MODELS['ollama']['model_name'], 
                    "prompt": prompt, 
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from model.")
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"An error occurred while generating the answer with Ollama: {str(e)}"
    
    def get_not_found_message(self, language: str) -> str:
        """Get 'not found' message in specified language"""
        lang_config = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["english"])
        return lang_config['not_found']
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if specified model is available"""
        if model_name == "gemini":
            return GEMINI_API_KEY is not None
        elif model_name == "ollama":
            try:
                response = requests.get("http://localhost:11434/api/tags")
                return response.status_code == 200
            except:
                return False
        return False 