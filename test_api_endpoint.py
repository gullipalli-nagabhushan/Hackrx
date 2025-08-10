#!/usr/bin/env python3
"""
Test script to test the API endpoint comprehensively
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict

async def test_api_endpoint():
    """Test the API endpoint with various scenarios"""
    
    base_url = "http://localhost:8000"
    auth_token = "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing API Endpoint")
    print("=" * 50)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Single Question Test",
            "data": {
                "documents": "https://example.com/test.pdf",
                "questions": ["What is the grace period for premium payment?"]
            }
        },
        {
            "name": "Multiple Questions Test",
            "data": {
                "documents": "https://example.com/test.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the waiting periods for pre-existing diseases?",
                    "Does this policy cover maternity expenses?"
                ]
            }
        },
        {
            "name": "Many Questions Test (10 questions)",
            "data": {
                "documents": "https://example.com/test.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the waiting periods for pre-existing diseases?",
                    "Does this policy cover maternity expenses?",
                    "What is covered under this policy?",
                    "What are the exclusions?",
                    "How long is the waiting period?",
                    "What documents are required?",
                    "What is the claim process?",
                    "What is the premium amount?",
                    "What is the sum insured?"
                ]
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Test {i}: {scenario['name']} ---")
            
            try:
                start_time = time.time()
                
                async with session.post(
                    f"{base_url}/api/v1/hackrx/run",
                    headers=headers,
                    json=scenario['data'],
                    timeout=aiohttp.ClientTimeout(total=60)
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
                            print(f"   ⚠️  Found {len(busy_errors)} 'Service is busy' errors")
                            for j, error in enumerate(busy_errors, 1):
                                print(f"      Error {j}: {error}")
                        else:
                            print(f"   ✅ No 'Service is busy' errors found")
                        
                        # Show first answer as sample
                        if answers:
                            print(f"   Sample Answer: {answers[0][:100]}...")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error: {error_text}")
                        
            except asyncio.TimeoutError:
                print(f"   ❌ Timeout after 60 seconds")
            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")
            
            # Wait between tests
            await asyncio.sleep(2)

async def test_rate_limiting():
    """Test rate limiting by making multiple rapid requests"""
    
    print("\n🧪 Testing Rate Limiting")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    auth_token = "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["What is the grace period for premium payment?"]
    }
    
    async with aiohttp.ClientSession() as session:
        print("Making 5 rapid requests...")
        
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                session.post(
                    f"{base_url}/api/v1/hackrx/run",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                )
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses, 1):
            print(f"\n   Request {i}:")
            if isinstance(response, Exception):
                print(f"   ❌ Exception: {str(response)}")
            else:
                try:
                    result = await response.json()
                    status = response.status
                    answers = result.get('answers', [])
                    
                    print(f"   Status: {status}")
                    print(f"   Answers: {len(answers)}")
                    
                    if answers and "Service is busy" in answers[0]:
                        print(f"   ⚠️  Got 'Service is busy' error")
                    else:
                        print(f"   ✅ Success")
                        
                except Exception as e:
                    print(f"   ❌ Error parsing response: {str(e)}")

async def main():
    """Run all tests"""
    print("🚀 API Endpoint Comprehensive Test")
    print("=" * 60)
    
    # Test normal scenarios
    await test_api_endpoint()
    
    # Test rate limiting
    await test_rate_limiting()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("If you're getting 'Service is busy' errors:")
    print("1. Check your network connection")
    print("2. Don't make too many requests too quickly")
    print("3. Check if your API keys have usage limits")
    print("4. Try again in a few minutes")

if __name__ == "__main__":
    asyncio.run(main())
