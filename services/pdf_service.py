"""
PDF service for processing and extracting text from PDF documents
"""
from pdfplumber import open as pdf_open
from typing import Tuple, List
import os


class PDFService:
    """Handles PDF processing and text extraction"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text from PDF
        """
        all_text = ""
        try:
            with pdf_open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    all_text += f"\n\n--- Page {i + 1} ---\n{page_text or ''}"
            return all_text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text_from_pdf_file(self, file_path: str) -> str:
        """
        Extract text from PDF file path
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text from PDF
        """
        if not os.path.exists(file_path):
            print(f"PDF file not found: {file_path}")
            return ""
        
        all_text = ""
        try:
            with pdf_open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    all_text += f"\n\n--- Page {i + 1} ---\n{page_text or ''}"
            return all_text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def get_pdf_info(self, uploaded_file) -> dict:
        """
        Get basic information about the PDF
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with PDF information
        """
        try:
            with pdf_open(uploaded_file) as pdf:
                info = {
                    "num_pages": len(pdf.pages),
                    "file_size": uploaded_file.size,
                    "file_name": uploaded_file.name
                }
            return info
        except Exception as e:
            print(f"Error getting PDF info: {str(e)}")
            return {}
    
    def validate_pdf(self, uploaded_file) -> bool:
        """
        Validate if uploaded file is a valid PDF
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            # Check file extension
            if not uploaded_file.name.lower().endswith('.pdf'):
                return False
            
            # Try to open with pdfplumber
            with pdf_open(uploaded_file) as pdf:
                if len(pdf.pages) == 0:
                    return False
            return True
        except Exception:
            return False 