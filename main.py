# main.py - FastAPI Main Application
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import uvicorn
from datetime import datetime
import logging
import os

# Import custom modules
from document_processor import DocumentProcessor
from query_engine import QueryEngine
from vector_store import PineconeVectorStore
from database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document processing and query system for insurance, legal, HR, and compliance domains",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
VALID_TOKEN = "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Pydantic models for request/response
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL or path to document(s)")
    questions: List[str] = Field(..., description="List of natural language queries")

class QueryResponse(BaseModel):
    answers: List[str]

# Initialize system components
document_processor = DocumentProcessor()
vector_store = PineconeVectorStore()
query_engine = QueryEngine()
database_manager = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    logger.info("Starting LLM-Powered Query-Retrieval System...")
    await vector_store.initialize()
    await query_engine.initialize()
    await database_manager.initialize()
    logger.info("System initialization completed")

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing document queries
    """
    try:
        start_time = datetime.now()

        # Step 1: Process and ingest document
        logger.info(f"Processing document: {request.documents}")
        document_chunks = await document_processor.process_document(request.documents)

        # Step 2: Generate embeddings and store in vector database
        await vector_store.add_documents(document_chunks)

        # Step 3: Process each query
        answers = []

        for question in request.questions:
            logger.info(f"Processing query: {question}")

            # Parse query intent and generate embedding
            query_embedding = await query_engine.generate_query_embedding(question)

            # Perform semantic search
            relevant_chunks = await vector_store.similarity_search(
                query_embedding, 
                top_k=5
            )

            # Generate answer using LLM
            answer = await query_engine.generate_answer(
                question, 
                relevant_chunks
            )

            answers.append(answer)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {processing_time:.2f} seconds")

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
