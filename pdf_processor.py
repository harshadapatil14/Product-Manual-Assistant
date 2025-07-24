# pdf_processor_plumber.py

import pdfplumber
import os
from PIL import Image
import pytesseract
from typing import List, Tuple


class PDFPlumberProcessor:
    def __init__(self, pdf_path: str, image_output_dir: str = "extracted_images"):
        self.pdf_path = pdf_path
        self.image_output_dir = image_output_dir
        os.makedirs(self.image_output_dir, exist_ok=True)

    def extract_text(self) -> str:
        """Extract text from PDF using pdfplumber."""
        all_text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                all_text += f"\n\n--- Page {i + 1} ---\n{page_text or ''}"
        return all_text

    def extract_images(self) -> List[str]:
        """Extract images from each PDF page using pdfplumber."""
        image_paths = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try to extract the entire page as an image
                im = page.to_image(resolution=300)
                image_path = os.path.join(self.image_output_dir, f"page_{i+1}.png")
                im.save(image_path, format="PNG")
                image_paths.append(image_path)
        return image_paths

    def perform_ocr_on_images(self, image_paths: List[str]) -> str:
        """Run OCR on each saved image and return combined text."""
        ocr_text = ""
        for path in image_paths:
            try:
                image = Image.open(path)
                text = pytesseract.image_to_string(image)
                ocr_text += f"\n\n--- OCR from {path} ---\n{text}"
            except Exception as e:
                print(f"[!] OCR failed on {path}: {e}")
        return ocr_text

    def process_pdf(self) -> Tuple[str, str]:
        """Complete processing: extract raw text, images, and OCR text."""
        print("[*] Extracting text from PDF...")
        raw_text = self.extract_text()

        print("[*] Saving page images for OCR...")
        image_paths = self.extract_images()

        print("[*] Performing OCR on images...")
        ocr_text = self.perform_ocr_on_images(image_paths)

        full_text = raw_text + "\n\n" + ocr_text
        return full_text, raw_text


# Example usage
if __name__ == "__main__":
    processor = PDFPlumberProcessor("R6400_UM.pdf")
    full_text, raw_text = processor.process_pdf()

    with open("full_text_plumber.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    print("\n[✓] PDF processing done with pdfplumber. Output saved to full_text_plumber.txt")
