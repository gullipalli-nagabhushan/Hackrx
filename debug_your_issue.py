#!/usr/bin/env python3
"""
Diagnostic tool to help identify "Service is busy" issues
"""

import asyncio
import aiohttp
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_your_specific_request():
    """Test with your specific request data"""
    
    print("🔍 Diagnostic Tool for 'Service is busy' Issues")
    print("=" * 60)
    
    # Get your specific request data
    print("Please provide your request details:")
    
    # You can modify these values to match your actual request
    document_url = input("Document URL (or press Enter for default): ").strip()
    if not document_url:
        document_url = "https://example.com/test.pdf"
    
    questions_input = input("Questions (comma-separated, or press Enter for default): ").strip()
    if not questions_input:
        questions = ["What is the grace period for premium payment?"]
    else:
        questions = [q.strip() for q in questions_input.split(",")]
    
    print(f"\n📋 Testing with:")
    print(f"   Document: {document_url}")
    print(f"   Questions: {questions}")
    
    # Test the request
    base_url = "http://localhost:8000"
    auth_token = "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "documents": document_url,
        "questions": questions
    }
    
    print(f"\n🚀 Making request...")
    
    try:
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/v1/hackrx/run",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=120)  # 2 minutes timeout
            ) as response:
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                print(f"   Status Code: {response.status}")
                print(f"   Response Time: {response_time:.2f}ms")
                
                if response.status == 200:
                    result = await response.json()
                    answers = result.get('answers', [])
                    
                    print(f"   ✅ Success! Got {len(answers)} answers")
                    
                    # Check for "Service is busy" errors
                    busy_errors = [ans for ans in answers if "Service is busy" in ans]
                    if busy_errors:
                        print(f"   ⚠️  Found {len(busy_errors)} 'Service is busy' errors:")
                        for i, error in enumerate(busy_errors, 1):
                            print(f"      Error {i}: {error}")
                    else:
                        print(f"   ✅ No 'Service is busy' errors found")
                    
                    # Show all answers
                    for i, answer in enumerate(answers, 1):
                        print(f"   Answer {i}: {answer}")
                        
                else:
                    error_text = await response.text()
                    print(f"   ❌ Error: {error_text}")
                    
    except asyncio.TimeoutError:
        print(f"   ❌ Timeout after 120 seconds")
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")

async def check_api_keys():
    """Check if API keys are working"""
    
    print("\n🔑 Checking API Keys...")
    print("=" * 40)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print(f"OpenAI API Key: {'✅ Set' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Not set'}")
    print(f"Groq API Key: {'✅ Set' if groq_key and groq_key != 'your_groq_api_key_here' else '❌ Not set'}")
    
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("⚠️  OpenAI API key not configured - embeddings will fail")
    
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("⚠️  Groq API key not configured - LLM will fail")

async def check_system_status():
    """Check if the system is running"""
    
    print("\n🏥 Checking System Status...")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ System is running and healthy")
                    print(f"   Status: {result.get('status', 'unknown')}")
                else:
                    print(f"❌ System health check failed: {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to system: {str(e)}")
        print("   Make sure the application is running with: python run.py")

async def main():
    """Run diagnostics"""
    
    # Check system status
    await check_system_status()
    
    # Check API keys
    await check_api_keys()
    
    # Test your specific request
    await test_your_specific_request()
    
    print("\n" + "=" * 60)
    print("📊 Diagnostic Summary:")
    print("If you're still getting 'Service is busy' errors:")
    print("1. Check the document URL is accessible")
    print("2. Try with fewer questions first")
    print("3. Check your API usage limits")
    print("4. Wait a few minutes and try again")
    print("5. Check the application logs: tail -f app.log")

if __name__ == "__main__":
    asyncio.run(main())
