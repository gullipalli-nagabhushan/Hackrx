# query_engine.py - Query Processing and Answer Generation Engine
import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import json
import hashlib
import time
import random
from config import settings
from groq_status_monitor import get_groq_status, should_retry_groq, get_groq_retry_delay, get_fallback_answer

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self):
        self.openai_client = None
        self.groq_llm = None
        self.embedding_cache = {}
        self.answer_cache = {}
        self.precomputed_answers = {}  # Pre-computed answers for common questions
        self.fast_cache = {}  # Ultra-fast in-memory cache
        self.instant_cache = {}  # Instant cache for <20ms responses
        self.response_cache = {}  # Complete response cache
        self.retry_attempts = 3  # Number of retry attempts for API calls
        self.base_delay = 1.0  # Base delay for exponential backoff
        self.max_delay = 10.0  # Maximum delay between retries

    async def initialize(self):
        """Initialize OpenAI and Groq clients"""
        # Check if we have valid API keys
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            logger.warning("OpenAI API key not set or is placeholder - using mock mode for embeddings")
            self.openai_client = None
        else:
            self.openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )
        
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
            logger.warning("Groq API key not set or is placeholder - using mock mode for LLM")
            self.groq_llm = None
        else:
            self.groq_llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.1,  # STRICTLY deterministic
                max_tokens=200,  # Reduced for concise answers
            )
        
        logger.info("Query engine initialized with OpenAI (embeddings) and Langchain Groq (LLM)")

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff for handling API errors"""
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                if any(keyword in error_str for keyword in ['503', 'service unavailable', 'timeout', 'rate limit', '429']):
                    if attempt < self.retry_attempts - 1:  # Don't sleep on last attempt
                        delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                        logger.warning(f"API call failed (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}. Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-retryable error, break immediately
                    break
        
        # If we get here, all retries failed
        raise last_exception

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query with instant caching using OpenAI"""
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
            
            # Generate embedding with retry logic
            async def _generate_embedding():
                return await asyncio.wait_for(
                    self.openai_client.embeddings.create(
                        model=settings.OPENAI_EMBEDDING_MODEL,
                        input=query.strip()
                    ),
                    timeout=10.0
                )
            
            response = await self._retry_with_backoff(_generate_embedding)
            embedding = response.data[0].embedding
            
            # Cache the embedding in all layers
            self.instant_cache[cache_key] = embedding
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            raise

    async def generate_answer(self, question: str, relevant_chunks: List[Dict[str, Any]], document_url: str = None) -> str:
        """Generate instant answer using Llama70B with ultra-aggressive caching and robust error handling"""
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

            # Check if Groq API key is available
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
                logger.error("Groq API key not configured. Cannot generate answers without API access.")
                return "Error: Groq API key not configured. Please set GROQ_API_KEY environment variable to use this service."

            # Prepare comprehensive context from all relevant chunks
            context = ""
            for i, chunk in enumerate(relevant_chunks[:15]):  # Use up to 15 chunks for better coverage
                context += f"{chunk['content']}\n\n"
            
            if not context.strip():
                context = "No relevant information found in the documents."
            
            # Enhanced prompt template with document URL for 100% accuracy and clean answers
            document_info = f"Source Document: {document_url}" if document_url else "Source Document: Not specified"
            prompt = f"""Question: {question}

{document_info}

Document :
{context}

CRITICAL INSTRUCTIONS: 
1. Answer based on the document content above and don't use otherthan that 
2. MAXIMUM 2 sentences - this is STRICTLY enforced
3. Be extremely concise and direct
4. ALWAYS provide an answer based on the document sections if any relevant information exists
5. Provide clean, professional answers like: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
6. IMPORTANT: Keep answers under 300 characters and maximum 2 sentences
7. Do not use your own knowledge to answer the question, only use the document content and the document url"""

            # Generate answer with retry logic using Langchain Groq
            async def _generate_groq_response():
                messages = [
                    SystemMessage(content="You are a specialized document analysis assistant. Your role is to provide accurate, document-specific answers based on the provided document sections. CRITICAL: Keep answers to MAXIMUM 2 sentences and under 300 characters - this is STRICTLY enforced. ALWAYS use the information from the document sections to answer questions when relevant information exists. Provide clean, concise, professional answers. IMPORTANT: Be extremely concise while maintaining accuracy and using available document information."),
                    HumanMessage(content=prompt)
                ]
                
                return await asyncio.wait_for(
                    self.groq_llm.ainvoke(messages),
                    timeout=15.0
                )

            response = await self._retry_with_backoff(_generate_groq_response)
            answer = response.content.strip()
            
            # Cache the answer in all layers
            self.instant_cache[cache_key] = answer
            self.fast_cache[cache_key] = answer
            self.answer_cache[cache_key] = answer
            return answer

        except Exception as e:
            logger.error(f"❌ Answer generation failed: {str(e)}")
            
            # Check Groq service status for better error handling
            try:
                groq_status = await get_groq_status()
                if not groq_status.get("is_healthy", True):
                    logger.warning(f"Groq service appears to be unhealthy: {groq_status.get('recommendation', 'Unknown issue')}")
                    
                    # Try to generate a fallback response using available context
                    context = ""
                    for chunk in relevant_chunks[:5]:  # Use first 5 chunks for fallback
                        context += f"{chunk['content']}\n\n"
                    
                    fallback_answer = await get_fallback_answer(question, context)
                    return fallback_answer
            except Exception as status_error:
                logger.error(f"Error checking Groq status: {str(status_error)}")
            
            # Provide more specific error messages based on error type
            error_str = str(e).lower()
            
            if "503" in error_str or "service unavailable" in error_str:
                return "Groq service is temporarily unavailable. Please try again in a few minutes. Visit https://groqstatus.com/ for service status."
            elif "timeout" in error_str:
                return "Request timed out. Please try again with a simpler question."
            elif "rate limit" in error_str or "429" in error_str:
                return "Service is busy. Please wait a moment and try again."
            elif "authentication" in error_str or "api key" in error_str:
                return "Authentication error. Please check API configuration."
            elif "quota" in error_str or "billing" in error_str:
                return "API quota exceeded. Please check your Groq account billing status."
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