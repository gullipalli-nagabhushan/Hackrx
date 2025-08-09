# query_engine.py - Query Processing and Answer Generation Engine
import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
import json
import hashlib
import time
from config import settings

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self):
        self.openai_client = None
        self.embedding_cache = {}
        self.answer_cache = {}
        self.precomputed_answers = {}  # Pre-computed answers for common questions
        self.fast_cache = {}  # Ultra-fast in-memory cache
        self.instant_cache = {}  # Instant cache for <20ms responses
        self.response_cache = {}  # Complete response cache

    async def initialize(self):
        """Initialize OpenAI client"""
        # Check if we have valid API keys
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            logger.warning("OpenAI API key not set or is placeholder - using mock mode for testing")
            self.openai_client = None
            return
        
        self.openai_client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        logger.info("Query engine initialized")

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query with instant caching"""
        try:
            if not query or not query.strip():
                raise ValueError("Empty query provided")
            
            # Instant cache check
            cache_key = hashlib.md5(query.strip().encode()).hexdigest()
            if cache_key in self.instant_cache:
                return self.instant_cache[cache_key]
            
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Check if OpenAI API key is available
            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
                logger.error("OpenAI API key not configured. Cannot generate embeddings without API access.")
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable to use this service.")
            
            # Generate embedding with reasonable timeout
            response = await asyncio.wait_for(
                self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=query.strip()
                ),
                timeout=10.0  # Increased timeout for reliability
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding in all layers
            self.instant_cache[cache_key] = embedding
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            raise

    async def generate_answer(self, question: str, relevant_chunks: List[Dict[str, Any]], document_url: str = None) -> str:
        """Generate instant answer using GPT-4o with ultra-aggressive caching"""
        try:
            if not question or not question.strip():
                return "No question provided."
            
            if not relevant_chunks:
                return "No relevant information found to answer this question."

            # Instant cache check with question hash
            question_hash = hashlib.md5(question.strip().encode()).hexdigest()
            cache_key = f"{question_hash}_{hash(str(relevant_chunks))}"
            
            # Check all cache layers for instant response
            if cache_key in self.instant_cache:
                return self.instant_cache[cache_key]
            
            if cache_key in self.fast_cache:
                return self.fast_cache[cache_key]
            
            if cache_key in self.answer_cache:
                return self.answer_cache[cache_key]

            # Check for pre-computed answers for common questions
            normalized_question = question.strip().lower()
            if normalized_question in self.precomputed_answers:
                answer = self.precomputed_answers[normalized_question]
                self.instant_cache[cache_key] = answer
                return answer

            # Check if OpenAI API key is available
            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
                logger.error("OpenAI API key not configured. Cannot generate answers without API access.")
                return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable to use this service."

            # Prepare comprehensive context from all relevant chunks
            context = ""
            for i, chunk in enumerate(relevant_chunks[:15]):  # Use up to 15 chunks for better coverage
                context += f"Document Section {i+1}: {chunk['content'][:1000]}\n\n"
            
            if not context.strip():
                context = "No relevant information found in the documents."
            # Enhanced prompt template with document URL for 100% accuracy and clean answers
            document_info = f"Source Document: {document_url}" if document_url else "Source Document: Not specified"
            prompt = f"""Question: {question}

{document_info}

Document Sections:
{context}

CRITICAL INSTRUCTIONS: 
1. Answer based on the document sections above
2. MAXIMUM 2 sentences - this is STRICTLY enforced
3. Be extremely concise and direct
4. If the document sections contain sufficient information, provide a clean answer
5. If the document sections don't contain enough information, use the source document URL to find the complete answer
6. If the question is not related to the document, provide a clear answer like: "The question is not related to the document."
6. Provide clean, professional answers like: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
7. IMPORTANT: Keep answers under 300 characters and maximum 2 sentences"""

            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a specialized document analysis assistant. Your role is to provide accurate, document-specific answers based on the provided document sections and the source document URL. CRITICAL: Keep answers to MAXIMUM 2 sentences and under 300 characters - this is STRICTLY enforced. If the document sections contain sufficient information, provide a clean, professional answer. If the document sections don't contain enough information, use the source document URL to find the complete answer. Provide clean, concise answers like: 'A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.' IMPORTANT: Be extremely concise while maintaining accuracy."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=200,  # Increased for comprehensive document access
                    timeout=15.0  # Increased timeout for reliability
                ),
                timeout=15.0
            )

            answer = response.choices[0].message.content.strip()
            
            # Cache the answer in all layers
            self.instant_cache[cache_key] = answer
            self.fast_cache[cache_key] = answer
            self.answer_cache[cache_key] = answer
            return answer

        except Exception as e:
            logger.error(f"❌ Answer generation failed: {str(e)}")
            
            # Provide more specific error messages
            if "timeout" in str(e).lower():
                return "Request timed out. Please try again with a simpler question."
            elif "rate limit" in str(e).lower():
                return "Service is busy. Please wait a moment and try again."
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                return "Authentication error. Please check API configuration."
            else:
                return f"Error processing question: {str(e)[:100]}..."

    async def generate_answers_batch(self, questions: List[str], all_relevant_chunks: List[List[Dict[str, Any]]], document_url: str = None) -> List[str]:
        """Generate answers for multiple questions in batch with instant parallel processing"""
        try:
            if len(questions) != len(all_relevant_chunks):
                raise ValueError("Number of questions must match number of chunk lists")
            
            # Process all answers in parallel with instant optimization
            tasks = []
            for question, relevant_chunks in zip(questions, all_relevant_chunks):
                task = self.generate_answer(question, relevant_chunks, document_url)
                tasks.append(task)
            
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"❌ Batch question {i+1} failed: {str(answer)}")
                    processed_answers.append("Error processing this question. Please try again.")
                else:
                    processed_answers.append(answer)
            
            return processed_answers
            
        except Exception as e:
            logger.error(f"❌ Batch answer generation failed: {str(e)}")
            raise