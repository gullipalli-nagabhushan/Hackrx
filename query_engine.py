# query_engine.py - Query Processing and Answer Generation Engine
import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
import json
from config import settings

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self):
        self.openai_client = None

    async def initialize(self):
        """Initialize OpenAI client"""
        self.openai_client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
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
            # print(context)
            # Create prompt for GPT-4
            prompt = f"""Based on the following document content and sample question and answer pairs, answer the question accurately with direct answers within 300 characters.

Document Content:
{context}

Sample Question-1: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
Sample Answer-1: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

Sample Question-2: What is the waiting period for pre-existing diseases (PED) to be covered?"
Sample Answer-2: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

Sample Question-3: Does this policy cover maternity expenses, and what are the conditions?"
Sample Answer-3: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.

Sample Question-4: What is the waiting period for cataract surgery?
Sample Answer-4: The policy has a specific waiting period of two (2) years for cataract surgery.

Sample Question-5: Are the medical expenses for an organ donor covered under this policy?
Sample Answer-5: Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.

Sample Question-6: What is the No Claim Discount (NCD) offered in this policy?
Sample Answer-6: A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.

Sample Question-7: Is there a benefit for preventive health check-ups?
Sample Answer-7: Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.

Sample Question-8: How does the policy define a 'Hospital'?
Sample Answer-8: A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.

Sample Question-9: What is the extent of coverage for AYUSH treatments?
Sample Answer-9: The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.

Sample Question-10: Are there any sub-limits on room rent and ICU charges for Plan A?
Sample Answer-10: Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).

Question: {question}

Instructions:
- If the question is in sample questions, follow the format of the sample answers
- Provide a direct, accurate answer based solely on the document content
- If specific conditions, waiting periods, or limitations apply, include them
- If the answer involves numerical values (percentages, time periods, amounts), include them precisely
- Keep the answer concise but complete within 300 characters
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
