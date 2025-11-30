import pdfplumber
import pytesseract
from PIL import Image
import io
from typing import List, Tuple
import re


class IngestionService:
    """Service for extracting text from PDFs and images using OCR."""

    @staticmethod
    def extract_from_pdf(file_path: str) -> Tuple[str, List[str]]:
        """Extract text from PDF file."""
        text_chunks = []
        full_text = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                        # Split into chunks (by sentences or paragraphs)
                        chunks = IngestionService._chunk_text(page_text)
                        text_chunks.extend(chunks)
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
        
        return full_text.strip(), text_chunks

    @staticmethod
    def extract_from_image(file_path: str) -> Tuple[str, List[str]]:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            chunks = IngestionService._chunk_text(text)
            return text.strip(), chunks
        except Exception as e:
            raise Exception(f"Error extracting image: {str(e)}")

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, file_type: str) -> Tuple[str, List[str]]:
        """Extract text from file bytes."""
        if file_type == "application/pdf":
            # Save to temp file or use in-memory PDF
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                return IngestionService.extract_from_pdf(tmp_path)
            finally:
                import os
                os.unlink(tmp_path)
        elif file_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
            chunks = IngestionService._chunk_text(text)
            return text.strip(), chunks
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    @staticmethod
    def cleanup_text(text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical symbols
        text = re.sub(r'[^\w\s\.\,\:\;\(\)\[\]\-\+\%\/]', '', text)
        return text.strip()

