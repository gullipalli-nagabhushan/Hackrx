# vector_store.py - Pinecone Vector Database Implementation
import asyncio
import logging
from typing import List, Dict, Any, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
from config import settings

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = "document-retrieval-index"
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.openai_client = None

    async def initialize(self):
        """Initialize Pinecone connection and index"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            # Initialize OpenAI for embeddings
            self.openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )

            # Create or connect to index
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")

            self.index = self.pc.Index(self.index_name)
            logger.info("Pinecone vector store initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

    async def add_documents(self, chunks: List[Any]):
        """Add document chunks to Pinecone index"""
        try:
            # Prepare texts for embedding
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings using OpenAI
            embeddings = await self._generate_embeddings(texts)

            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{chunk.metadata.get('source', 'unknown')}_{chunk.metadata.get('chunk_id', str(uuid.uuid4()))}"

                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": chunk.metadata.get('source', 'unknown'),
                        "chunk_id": chunk.metadata.get('chunk_id', str(uuid.uuid4())),
                        "total_pages": chunk.metadata.get('total_pages', 0),
                        "page_number": chunk.metadata.get('page_number', 1),
                        "document_type": chunk.metadata.get('document_type', 'unknown'),
                        "token_count": chunk.metadata.get('token_count', 0),
                        "content": chunk.content  # Include actual text content
                    }
                }
                vectors.append(vector)

            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Successfully added {len(vectors)} vectors to Pinecone")

        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            raise

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Split into smaller batches to avoid rate limits
            batch_size = 20
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch_texts
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Small delay to respect rate limits
                await asyncio.sleep(0.1)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search in Pinecone"""
        try:
            # Query Pinecone index
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )

            # Format results
            results = []
            for match in query_response.matches:
                result = {
                    'content': match.metadata.get('content', ''),
                    "metadata": {
                        **match.metadata,
                    },
                    'similarity_score': match.score,
                    'vector_id': match.id
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in Pinecone similarity search: {str(e)}")
            raise
