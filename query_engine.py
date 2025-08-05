# query_engine.py - Query Processing and Answer Generation Engine
import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
import json
import os

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self):
        self.openai_client = None

    async def initialize(self):
        """Initialize OpenAI client"""
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("Query engine initialized")

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    async def generate_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using GPT-4 based on relevant chunks"""
        try:
            # Prepare context from relevant chunks
            context = "\n\n".join([
                f"Content {i+1}: {chunk['content']}"
                for i, chunk in enumerate(relevant_chunks)
            ])

            # Create prompt for GPT-4
            prompt = f"""Based on the following document content, answer the question accurately and concisely.

Document Content:
{context}

Question: {question}

Instructions:
- Provide a direct, accurate answer based solely on the document content
- If specific conditions, waiting periods, or limitations apply, include them
- If the answer involves numerical values (percentages, time periods, amounts), include them precisely
- Keep the answer concise but complete
- If the information is not available in the document, state that clearly

Answer:"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert document analyst specializing in insurance policies, legal contracts, and compliance documents. Provide accurate, precise answers based strictly on the provided document content."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )

            answer = response.choices[0].message.content.strip()

            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
