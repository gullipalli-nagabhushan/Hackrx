# document_processor.py - Document Processing and Chunking
import asyncio
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import tempfile
import os
from pathlib import Path
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
from docx import Document as DocxDocument
import tiktoken
from config import settings
import io

logger = logging.getLogger(__name__)

class DocumentChunk:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.page_number = metadata.get('page_number')
        self.chunk_id = metadata.get('chunk_id')
        self.source = metadata.get('source')

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context preservation
            chunk_overlap=200,  # Increased for better context continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.session = None
        self.cache = {}  # Cache for processed documents

    async def initialize(self):
        """Initialize aiohttp session for document downloads"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)  # Ultra-aggressive timeout
            )

    async def process_document(self, document_url: str) -> List[Document]:
        """Process document with ultra-fast optimization"""
        try:
            # Check cache first
            cache_key = hashlib.md5(document_url.encode()).hexdigest()
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Download document
            content = await self._download_document(document_url)
            if not content:
                return []

            # Extract text based on file type
            text = await self._extract_text(content, document_url)
            if not text:
                return []

            # Split text into chunks
            chunks = await self._split_text(text, document_url)
            
            # Cache the result
            self.cache[cache_key] = chunks
            return chunks

        except Exception as e:
            logger.error(f"Error processing document {document_url}: {str(e)}")
            return []

    async def _download_document(self, url: str) -> Optional[bytes]:
        """Download document with ultra-fast optimization"""
        try:
            if not self.session:
                await self.initialize()

            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:  # Ultra-aggressive timeout
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to download document from {url}: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error downloading document from {url}: {str(e)}")
            return None

    async def _extract_text(self, content: bytes, url: str) -> Optional[str]:
        """Extract text from document with ultra-fast optimization"""
        try:
            file_extension = Path(url).suffix.lower()
            
            if file_extension == '.pdf':
                return await self._extract_pdf_text(content)
            elif file_extension in ['.docx', '.doc']:
                return await self._extract_docx_text(content)
            else:
                # Assume it's plain text
                return content.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Error extracting text from document: {str(e)}")
            return None

    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF with ultra-fast optimization"""
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            
            # Process all pages for complete coverage
            max_pages = len(pdf_reader.pages)
            for page_num in range(max_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text

        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""

    async def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX with ultra-fast optimization"""
        try:
            doc = DocxDocument(io.BytesIO(content))
            text = ""
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            return text

        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return ""

    async def _split_text(self, text: str, source: str) -> List[Document]:
        """Split text into chunks with ultra-fast optimization"""
        try:
            if not text.strip():
                return []

            # Use ultra-fast text splitting
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': source,
                        'chunk_id': f"{hashlib.md5(source.encode()).hexdigest()}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'document_type': Path(source).suffix.lower(),
                        'token_count': len(chunk.split())
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            return []

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    