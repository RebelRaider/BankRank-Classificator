import os
import tempfile
from typing import Optional, Dict, Any
import magic
import docx
from PyPDF2 import PdfReader
from ..core.config import settings
from ..utils.exceptions import FileProcessingError

class FileHandler:
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    def validate_file(self, filename: str, file_size: int) -> bool:
        """Validate uploaded file"""
        # Check file size
        if file_size > self.max_file_size:
            raise FileProcessingError(f"File size exceeds limit of {self.max_file_size} bytes")
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.allowed_extensions:
            raise FileProcessingError(f"File type {file_ext} not supported")
        
        return True
    
    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from different file formats"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext == '.txt':
                return self._extract_from_txt(file_content)
            elif file_ext == '.pdf':
                return self._extract_from_pdf(file_content)
            elif file_ext in ['.doc', '.docx']:
                return self._extract_from_docx(file_content)
            else:
                raise FileProcessingError(f"Unsupported file type: {file_ext}")
        
        except Exception as e:
            raise FileProcessingError(f"Error extracting text from {filename}: {str(e)}")
    
    def _extract_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file"""
        # Try different encodings
        encodings = ['utf-8', 'cp1251', 'koi8-r', 'iso-8859-5']
        
        for encoding in encodings:
            try:
                return file_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        raise FileProcessingError("Unable to decode text file with supported encodings")
    
    def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                reader = PdfReader(temp_file.name)
                text = ""
                
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
            
            finally:
                os.unlink(temp_file.name)
    
    def _extract_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                doc = docx.Document(temp_file.name)
                text_parts = []
                
                for paragraph in doc.paragraphs:
                    text_parts.append(paragraph.text)
                
                return "\n".join(text_parts).strip()
            
            finally:
                os.unlink(temp_file.name)
    
    def get_file_info(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Get file information"""
        file_size = len(file_content)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Detect MIME type
        mime_type = magic.from_buffer(file_content, mime=True)
        
        return {
            "filename": filename,
            "file_type": file_ext,
            "file_size": file_size,
            "mime_type": mime_type
        }
