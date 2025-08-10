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
import hashlib
import time
from config import settings

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = "document-retrieval-index"
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        self.executor = ThreadPoolExecutor(max_workers=16)  # Increased workers for instant speed
        self.openai_client = None
        self.session = None  # Ensure attribute exists for cleanup
        
        # Intelligent caching system to prevent stale results and question mixing
        self.embedding_cache = {}  # Cache for embeddings with TTL
        self.search_cache = {}  # Cache for search results with question fingerprinting
        self.document_cache = {}  # Cache for document-level results with TTL
        self.max_documents_cached = 10  # Maximum number of documents to cache
        self.cache_ttl = 300  # 5 minutes TTL to prevent stale results
        self.last_cache_cleanup = time.time()
        self.cache_cleanup_interval = 60  # Cleanup every minute

    async def initialize(self):
        """Initialize Pinecone connection and index"""
        try:
            # Check if we have valid API keys
            if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY == "your_pinecone_api_key_here":
                logger.warning("Pinecone API key not set or is placeholder - using mock mode for testing")
                self.pc = None
                self.index = None
                self.openai_client = None
                return
            
            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
                logger.warning("OpenAI API key not set or is placeholder - using mock mode for testing")
                self.pc = None
                self.index = None
                self.openai_client = None
                return
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            # Initialize OpenAI for embeddings
            self.openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )

            # Create or connect to index
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                await asyncio.sleep(5)
            else:
                logger.info(f"Connecting to existing Pinecone index: {self.index_name}")

            self.index = self.pc.Index(self.index_name)
            logger.info("Pinecone vector store initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            # Don't raise - allow system to continue in mock mode
            self.pc = None
            self.index = None
            self.openai_client = None

    async def add_documents(self, chunks: List[Any]):
        """Add document chunks to Pinecone index with instant batching"""
        try:
            if not chunks:
                logger.warning("No chunks provided to add to vector store")
                return

            # Normalize chunks to support both LangChain Document (page_content) and custom chunk (content)
            normalized_chunks: List[Dict[str, Any]] = []
            for chunk in chunks:
                text = getattr(chunk, "content", None)
                if text is None:
                    text = getattr(chunk, "page_content", None)
                metadata = getattr(chunk, "metadata", {}) or {}
                if text is None:
                    logger.warning("Skipping chunk with no text content")
                    continue
                if not isinstance(metadata, dict):
                    try:
                        metadata = dict(metadata)
                    except Exception:
                        metadata = {}
                normalized_chunks.append({"text": text, "metadata": metadata})

            if not normalized_chunks:
                logger.warning("No valid chunks with text content to index")
                return

            # Prepare texts for embedding in original order
            texts = [c["text"] for c in normalized_chunks]

            # Generate embeddings using OpenAI with intelligent caching
            embeddings = await self._generate_embeddings_batch(texts)

            # Prepare vectors for upsert
            vectors = []
            for normalized, embedding in zip(normalized_chunks, embeddings):
                source_value = normalized["metadata"].get("source", "unknown")
                chunk_id_value = normalized["metadata"].get("chunk_id", str(uuid.uuid4()))
                vector_id = f"{source_value}_{chunk_id_value}"
                
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "content": normalized["text"],
                        "source": source_value,
                        "chunk_id": chunk_id_value,
                        "created_at": time.time()
                    }
                }
                vectors.append(vector)

            # Upsert vectors in batches for instant speed
            if self.index:
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda b=batch: self.index.upsert(vectors=b)
                        ),
                        timeout=10.0
                    )
                logger.info(f"Successfully added {len(vectors)} document chunks to vector store")
            else:
                logger.warning("Pinecone index not available - skipping vector storage")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with intelligent caching"""
        try:
            if not texts:
                return []

            # Check cache for existing embeddings
            cached_embeddings = []
            texts_to_embed = []
            text_indices = []

            for i, text in enumerate(texts):
                # Create a more specific cache key including text content
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cache_key = f"embedding_{text_hash}"
                
                if cache_key in self.embedding_cache:
                    cache_entry = self.embedding_cache[cache_key]
                    # Check if cache entry is still valid
                    if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                        cached_embeddings.append((i, cache_entry['embedding']))
                    else:
                        # Remove expired cache entry
                        del self.embedding_cache[cache_key]
                        texts_to_embed.append(text)
                        text_indices.append(i)
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)

            # Generate embeddings for texts not in cache
            if texts_to_embed:
                if not self.openai_client:
                    logger.warning("OpenAI client not available - using mock embeddings")
                    new_embeddings = [np.random.rand(self.dimension).tolist() for _ in texts_to_embed]
                else:
                    try:
                        response = await asyncio.wait_for(
                            self.openai_client.embeddings.create(
                                input=texts_to_embed,
                                model=settings.OPENAI_EMBEDDING_MODEL
                            ),
                            timeout=10.0
                        )
                        new_embeddings = [embedding.embedding for embedding in response.data]
                    except Exception as e:
                        logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
                        # Fallback to mock embeddings
                        new_embeddings = [np.random.rand(self.dimension).tolist() for _ in texts_to_embed]

                # Cache new embeddings with timestamp
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    cache_key = f"embedding_{text_hash}"
                    self.embedding_cache[cache_key] = {
                        'embedding': embedding,
                        'timestamp': time.time()
                    }

                # Add new embeddings to results
                for i, embedding in zip(text_indices, new_embeddings):
                    cached_embeddings.append((i, embedding))

            # Sort by original index and return
            cached_embeddings.sort(key=lambda x: x[0])
            return [embedding for _, embedding in cached_embeddings]

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _create_question_fingerprint(self, query_text: str, query_embedding: List[float]) -> str:
        """Create a unique fingerprint for the question to prevent mixing"""
        # Combine text content and embedding hash for unique identification
        text_part = hashlib.md5(query_text.lower().strip().encode()).hexdigest()[:8]
        embedding_part = hashlib.md5(str(query_embedding[:10]).encode()).hexdigest()[:8]
        return f"{text_part}_{embedding_part}"
    
    def _create_document_fingerprint(self, document_url: str) -> str:
        """Create a unique fingerprint for the document"""
        # Create a hash based on document URL
        return hashlib.md5(document_url.encode()).hexdigest()[:12]
    
    def _create_cache_key(self, question_fingerprint: str, document_fingerprint: str) -> str:
        """Create a cache key combining question and document fingerprints"""
        return f"doc_{document_fingerprint}_q_{question_fingerprint}"

    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries to prevent memory bloat"""
        current_time = time.time()
        if current_time - self.last_cache_cleanup > self.cache_cleanup_interval:
            # Clean embedding cache
            expired_keys = [
                key for key, entry in self.embedding_cache.items()
                if current_time - entry['timestamp'] > self.cache_ttl
            ]
            for key in expired_keys:
                del self.embedding_cache[key]
            
            # Clean search cache
            expired_keys = [
                key for key, entry in self.search_cache.items()
                if current_time - entry['timestamp'] > self.cache_ttl
            ]
            for key in expired_keys:
                del self.search_cache[key]
            
            # Clean document cache and enforce document limit
            expired_keys = [
                key for key, entry in self.document_cache.items()
                if current_time - entry['timestamp'] > self.cache_ttl
            ]
            for key in expired_keys:
                del self.document_cache[key]
            
            # Enforce maximum document cache limit
            if len(self.document_cache) > self.max_documents_cached:
                # Remove oldest documents to stay within limit
                sorted_docs = sorted(
                    self.document_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                docs_to_remove = len(self.document_cache) - self.max_documents_cached
                for i in range(docs_to_remove):
                    key_to_remove = sorted_docs[i][0]
                    del self.document_cache[key_to_remove]
                logger.info(f"Removed {docs_to_remove} oldest documents to maintain cache limit")
            
            self.last_cache_cleanup = current_time
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def similarity_search(self, query_embedding: List[float], top_k: int = 5, query_text: str = "", document_url: str = "") -> List[Dict[str, Any]]:
        """Perform intelligent similarity search with question-specific caching"""
        try:
            if not query_embedding:
                logger.warning("Empty query embedding provided")
                return []

            # Clean up expired cache entries
            await self._cleanup_expired_cache()

            # Create unique question fingerprint to prevent mixing
            question_fingerprint = self._create_question_fingerprint(query_text, query_embedding)
            
            # Create document fingerprint if document_url is provided
            document_fingerprint = ""
            if document_url:
                document_fingerprint = self._create_document_fingerprint(document_url)
                logger.info(f"Document fingerprint: {document_fingerprint}")
            
            logger.info(f"Performing intelligent similarity search with fingerprint: {question_fingerprint}")
            logger.info(f"Pinecone API key status: {bool(settings.PINECONE_API_KEY)}")

            # Check document-level cache first (if document_url provided)
            if document_url and document_fingerprint:
                doc_cache_key = f"doc_{document_fingerprint}_q_{question_fingerprint}"
                if doc_cache_key in self.document_cache:
                    cache_entry = self.document_cache[doc_cache_key]
                    if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                        logger.info(f"Found valid document-level cache for: {document_fingerprint}")
                        return cache_entry['results']
                    else:
                        del self.document_cache[doc_cache_key]

            # Check regular search cache
            cache_key = f"search_{question_fingerprint}"
            if cache_key in self.search_cache:
                cache_entry = self.search_cache[cache_key]
                # Check if cache entry is still valid
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    logger.info(f"Found valid search results in cache for fingerprint: {question_fingerprint}")
                    return cache_entry['results']
                else:
                    # Remove expired cache entry
                    del self.search_cache[cache_key]

            # Check if Pinecone API key is available
            if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY == "your_pinecone_api_key_here":
                logger.error("Pinecone API key not configured. Cannot perform similarity search without API access.")
                return []

            # Ensure index is initialized
            if self.index is None:
                logger.error("Pinecone index not initialized. Initialize vector store before searching.")
                return []

            # Enhanced search: Get more results initially for better selection
            enhanced_top_k = min(top_k * 2, 20)  # Get more results for better selection
            
            # Build optional metadata filter to restrict results to the requested document only
            pinecone_filter = None
            if document_url:
                pinecone_filter = {"source": document_url}

            # Query Pinecone index with enhanced parameters and retries to mitigate transient timeouts
            attempt = 0
            query_response = None
            max_attempts = 3
            current_top_k = enhanced_top_k
            timeouts = [15.0, 25.0, 35.0]
            while attempt < max_attempts:
                try:
                    query_response = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: self.index.query(
                                vector=query_embedding,
                                top_k=current_top_k,
                                include_metadata=True,
                                include_values=False,
                                filter=pinecone_filter
                            )
                        ),
                        timeout=None  # No timeout for large indexes
                    )
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Similarity search attempt {attempt+1} timed out at {timeouts[min(attempt, len(timeouts)-1)]}s; "
                        f"retrying with adjusted parameters"
                    )
                    # Reduce load for next attempt
                    attempt += 1
                    if attempt == 1:
                        current_top_k = top_k
                    elif attempt >= 2:
                        current_top_k = max(5, min(top_k, 5))
                except Exception as inner_e:
                    # For non-timeout errors, log and break to outer handler
                    logger.error(f"Pinecone query failed on attempt {attempt+1}: {str(inner_e) or repr(inner_e)}")
                    raise

            if query_response is None:
                logger.error("Intelligent similarity search timed out after multiple attempts")
                return []

            # Format and enhance results with keyword relevance
            results = []
            for match in query_response.matches:
                content = match.metadata.get('content', '')
                
                # Calculate keyword relevance score
                keyword_score = 0
                if query_text:
                    query_words = query_text.lower().split()
                    content_lower = content.lower()
                    for word in query_words:
                        if len(word) > 3:  # Only consider meaningful words
                            if word in content_lower:
                                keyword_score += 1
                    
                    # Normalize keyword score
                    keyword_score = keyword_score / len(query_words) if query_words else 0
                
                # Combine semantic similarity with keyword relevance
                combined_score = (match.score * 0.7) + (keyword_score * 0.3)
                
                result = {
                    'content': content,
                    "metadata": {
                        **match.metadata,
                    },
                    'similarity_score': match.score,
                    'keyword_score': keyword_score,
                    'combined_score': combined_score,
                    'vector_id': match.id
                }
                results.append(result)

            # Sort by combined score and return top_k results
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            top_results = results[:top_k]

            # Cache results with question fingerprint and timestamp
            self.search_cache[cache_key] = {
                'results': top_results,
                'timestamp': time.time(),
                'fingerprint': question_fingerprint
            }
            
            # Also cache in document-level cache if document_url provided
            if document_url and document_fingerprint:
                doc_cache_key = f"doc_{document_fingerprint}_q_{question_fingerprint}"
                self.document_cache[doc_cache_key] = {
                    'results': top_results,
                    'timestamp': time.time(),
                    'fingerprint': question_fingerprint,
                    'document_fingerprint': document_fingerprint
                }
                logger.info(f"Cached results for document: {document_fingerprint}")
            
            return top_results

        except asyncio.TimeoutError:
            logger.error("Error in intelligent similarity search: request timed out")
            return []
        except Exception as e:
            logger.error(f"Error in intelligent similarity search: {str(e) or repr(e)}")
            return []

    async def delete_vectors_by_source(self, document_url: str) -> None:
        """Delete all vectors in the index that belong to a specific document URL."""
        try:
            if not self.index:
                logger.warning("Pinecone index not available - skipping vector deletion")
                return
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.delete(filter={"source": document_url})
            )
            logger.info(f"Deleted vectors for document source: {document_url}")
        except Exception as e:
            logger.error(f"Error deleting vectors for source {document_url}: {str(e)}")

    async def reset_index(self) -> None:
        """Delete all vectors from the index for a clean start."""
        try:
            if not self.pc or not self.index:
                logger.warning("Pinecone not initialized - skipping index reset")
                return
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.delete(delete_all=True)
            )
            logger.info("Pinecone index cleared (delete_all=True)")
        except Exception as e:
            logger.error(f"Error resetting Pinecone index: {str(e)}")

    async def clear_cache(self):
        """Clear all caches - useful for testing or when documents are updated"""
        self.embedding_cache.clear()
        self.search_cache.clear()
        self.document_cache.clear()
        logger.info("All caches cleared")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            'embedding_cache_size': len(self.embedding_cache),
            'search_cache_size': len(self.search_cache),
            'document_cache_size': len(self.document_cache),
            'max_documents_cached': self.max_documents_cached,
            'cache_ttl': self.cache_ttl,
            'last_cleanup': self.last_cache_cleanup
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.session:
            await self.session.close()
