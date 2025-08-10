#!/usr/bin/env python3
"""
Debug script to test OpenAI and Groq APIs directly
"""

import asyncio
import os
from dotenv import load_dotenv
import openai
import groq
import time

# Load environment variables
load_dotenv()

async def test_openai_embeddings():
    """Test OpenAI embeddings API"""
    print("🔍 Testing OpenAI Embeddings API...")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured")
        return False
    
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Test embedding generation
        start_time = time.time()
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input="Test query for embeddings"
        )
        end_time = time.time()
        
        if response.data and len(response.data) > 0:
            embedding = response.data[0].embedding
            print(f"✅ OpenAI embeddings working")
            print(f"   Response time: {(end_time - start_time)*1000:.2f}ms")
            print(f"   Embedding dimension: {len(embedding)}")
            return True
        else:
            print("❌ OpenAI embeddings failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI embeddings error: {str(e)}")
        return False

async def test_groq_llm():
    """Test Groq LLM API"""
    print("\n🔍 Testing Groq LLM API...")
    print("=" * 50)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("❌ Groq API key not configured")
        return False
    
    try:
        client = groq.AsyncGroq(api_key=api_key)
        
        # Test chat completion
        start_time = time.time()
        response = await client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep responses short and concise."
                },
                {
                    "role": "user",
                    "content": "What is 2+2? Answer in one word."
                }
            ],
            temperature=0.1,
            max_tokens=10
        )
        end_time = time.time()
        
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content.strip()
            print(f"✅ Groq LLM working")
            print(f"   Response time: {(end_time - start_time)*1000:.2f}ms")
            print(f"   Answer: {answer}")
            return True
        else:
            print("❌ Groq LLM failed - no choices returned")
            return False
            
    except Exception as e:
        print(f"❌ Groq LLM error: {str(e)}")
        return False

async def test_rate_limits():
    """Test rate limits by making multiple requests"""
    print("\n🔍 Testing Rate Limits...")
    print("=" * 50)
    
    # Test OpenAI rate limits
    print("Testing OpenAI rate limits...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        try:
            client = openai.AsyncOpenAI(api_key=openai_key)
            
            # Make 3 quick requests
            for i in range(3):
                start_time = time.time()
                response = await client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=f"Test query {i+1}"
                )
                end_time = time.time()
                print(f"   Request {i+1}: {(end_time - start_time)*1000:.2f}ms")
                await asyncio.sleep(0.1)  # Small delay
                
            print("✅ OpenAI rate limit test passed")
        except Exception as e:
            print(f"❌ OpenAI rate limit test failed: {str(e)}")
    
    # Test Groq rate limits
    print("\nTesting Groq rate limits...")
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            client = groq.AsyncGroq(api_key=groq_key)
            
            # Make 3 quick requests
            for i in range(3):
                start_time = time.time()
                response = await client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "user", "content": f"Say 'test {i+1}'"}
                    ],
                    max_tokens=5
                )
                end_time = time.time()
                answer = response.choices[0].message.content.strip()
                print(f"   Request {i+1}: {(end_time - start_time)*1000:.2f}ms - {answer}")
                await asyncio.sleep(0.1)  # Small delay
                
            print("✅ Groq rate limit test passed")
        except Exception as e:
            print(f"❌ Groq rate limit test failed: {str(e)}")

async def main():
    """Run all tests"""
    print("🚀 API Diagnostic Tool")
    print("=" * 60)
    
    # Test individual APIs
    openai_ok = await test_openai_embeddings()
    groq_ok = await test_groq_llm()
    
    # Test rate limits
    await test_rate_limits()
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   OpenAI Embeddings: {'✅ Working' if openai_ok else '❌ Failed'}")
    print(f"   Groq LLM: {'✅ Working' if groq_ok else '❌ Failed'}")
    
    if not openai_ok or not groq_ok:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Check your API keys are valid")
        print("2. Check your API usage limits")
        print("3. Check your internet connection")
        print("4. Try again in a few minutes if rate limited")

if __name__ == "__main__":
    asyncio.run(main())
