# document_processor.py - Document Processing Module
import asyncio
import aiohttp
import io
from typing import List, Dict, Any
import PyPDF2
import docx
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

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
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    async def process_document(self, document_url: str) -> List[DocumentChunk]:
        """
        Process document from URL and return chunks
        """
        try:
            # Download document
            document_content = await self._download_document(document_url)

            # Determine document type and extract text
            if document_url.lower().endswith('.pdf'):
                text, metadata = await self._process_pdf(document_content)
            elif document_url.lower().endswith('.docx'):
                text, metadata = await self._process_docx(document_content)
            else:
                # Default to PDF processing
                text, metadata = await self._process_pdf(document_content)

            # Create chunks
            chunks = await self._create_chunks(text, metadata, document_url)

            logger.info(f"Processed {len(chunks)} chunks from {document_url}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing document {document_url}: {str(e)}")
            raise

    async def _download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise Exception(f"Failed to download document: {response.status}")
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise

    async def _process_pdf(self, content: bytes) -> tuple[str, Dict[str, Any]]:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            page_metadata = []

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n\nPage {i+1}:\n{page_text}"
                page_metadata.append({
                    'page_number': i+1,
                    'page_text_length': len(page_text)
                })

            metadata = {
                'total_pages': len(reader.pages),
                'page_metadata': page_metadata,
                'document_type': 'pdf'
            }

            return text, metadata
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    async def _process_docx(self, content: bytes) -> tuple[str, Dict[str, Any]]:
        """Extract text from DOCX"""
        try:
            doc_file = io.BytesIO(content)
            doc = docx.Document(doc_file)

            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    text += f"Table: {row_text}\n"

            metadata = {
                'paragraphs_count': len(doc.paragraphs),
                'tables_count': len(doc.tables),
                'document_type': 'docx'
            }

            return text, metadata
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise

    async def _create_chunks(self, text: str, metadata: Dict[str, Any], source: str) -> List[DocumentChunk]:
        """Create semantic chunks from text"""
        try:
            # Split text into chunks
            docs = self.text_splitter.create_documents([text])

            chunks = []
            for i, doc in enumerate(docs):
                chunk_metadata = {
                    **metadata,
                    'chunk_id': f"chunk_{i}",
                    'source': source,
                    'token_count': len(self.tokenizer.encode(doc.page_content))
                }

                chunk = DocumentChunk(
                    content=doc.page_content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise
