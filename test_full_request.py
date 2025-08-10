#!/usr/bin/env python3
"""
Test script to simulate the full request flow
"""

import asyncio
import os
from dotenv import load_dotenv
from query_engine import QueryEngine
from vector_store import PineconeVectorStore
import time

# Load environment variables
load_dotenv()

async def test_full_request():
    """Test the full request flow"""
    
    print("🧪 Testing Full Request Flow")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing components...")
    query_engine = QueryEngine()
    vector_store = PineconeVectorStore()
    
    await query_engine.initialize()
    await vector_store.initialize()
    print("✅ Components initialized")
    
    # Test data
    test_questions = [
        "What is the grace period for premium payment?",
        "What are the waiting periods for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
    
    test_document_url = "https://example.com/test-document.pdf"
    
    print(f"\n2. Testing with {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        try:
            # Step 1: Generate embedding
            print("   Step 1: Generating embedding...")
            start_time = time.time()
            embedding = await query_engine.generate_query_embedding(question)
            embedding_time = (time.time() - start_time) * 1000
            print(f"   ✅ Embedding generated in {embedding_time:.2f}ms")
            
            # Step 2: Search vector store
            print("   Step 2: Searching vector store...")
            start_time = time.time()
            relevant_chunks = await vector_store.similarity_search(
                embedding, 
                top_k=5, 
                query_text=question, 
                document_url=test_document_url
            )
            search_time = (time.time() - start_time) * 1000
            print(f"   ✅ Found {len(relevant_chunks)} relevant chunks in {search_time:.2f}ms")
            
            # Step 3: Generate answer
            print("   Step 3: Generating answer...")
            start_time = time.time()
            answer = await query_engine.generate_answer(question, relevant_chunks, test_document_url)
            answer_time = (time.time() - start_time) * 1000
            print(f"   ✅ Answer generated in {answer_time:.2f}ms")
            print(f"   Answer: {answer}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")

async def test_with_mock_data():
    """Test with mock document chunks"""
    
    print("\n🧪 Testing with Mock Data")
    print("=" * 50)
    
    # Initialize components
    query_engine = QueryEngine()
    await query_engine.initialize()
    
    # Mock document chunks
    mock_chunks = [
        {
            "content": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "metadata": {"source": "test_document"}
        },
        {
            "content": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "metadata": {"source": "test_document"}
        },
        {
            "content": "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.",
            "metadata": {"source": "test_document"}
        }
    ]
    
    test_questions = [
        "What is the grace period for premium payment?",
        "What are the waiting periods for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
    
    print(f"Testing with {len(test_questions)} questions and {len(mock_chunks)} mock chunks...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        try:
            start_time = time.time()
            answer = await query_engine.generate_answer(question, mock_chunks, "test_document.pdf")
            answer_time = (time.time() - start_time) * 1000
            print(f"✅ Answer generated in {answer_time:.2f}ms")
            print(f"Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")

async def main():
    """Run all tests"""
    print("🚀 Full Request Flow Test")
    print("=" * 60)
    
    # Test with mock data first (should work)
    await test_with_mock_data()
    
    # Test with real vector store (might fail)
    await test_full_request()

if __name__ == "__main__":
    asyncio.run(main())
