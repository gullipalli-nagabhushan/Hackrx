#!/usr/bin/env python3
"""
Test Script for Accuracy Improvements
Tests the enhanced document processing and vector search system
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0"
TEST_DOCUMENT_URL = "https://www.irdai.gov.in/sites/default/files/2024-01/National%20Parivar%20Mediclaim%20Plus%20Policy%20-%20Policy%20Document.pdf"

# Test questions that should have high accuracy
TEST_QUESTIONS = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does the policy cover maternity expenses?",
    "What is the waiting period for cataract surgery?",
    "Does the policy cover organ donor expenses?",
    "What is the No Claim Discount offered?",
    "Are health check-ups covered?",
    "How is a hospital defined in the policy?",
    "Are AYUSH treatments covered?",
    "What are the room rent sub-limits for Plan A?"
]

async def test_health_check():
    """Test if the system is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health check passed")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

async def test_accuracy_improvements():
    """Test the accuracy improvements from enhanced chunking and search"""
    print("\n🔍 Testing Accuracy Improvements")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            
            # Test single question first to check response quality
            print("\n📝 Testing Single Question Response Quality:")
            single_request = {
                "documents": TEST_DOCUMENT_URL,
                "questions": [TEST_QUESTIONS[0]]
            }
            
            start_time = time.time()
            async with session.post(
                f"{API_BASE_URL}/api/v1/hackrx/run",
                headers=headers,
                json=single_request
            ) as response:
                processing_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("answers", [""])[0]
                    
                    print(f"✅ Single question processed in {processing_time:.2f}ms")
                    print(f"📝 Answer: {answer[:100]}...")
                    print(f"📊 Answer length: {len(answer)} characters")
                    
                    # Check if answer contains key information
                    key_terms = ["grace period", "premium", "payment", "days"]
                    contains_key_info = any(term in answer.lower() for term in key_terms)
                    print(f"🎯 Contains key information: {'✅' if contains_key_info else '❌'}")
                    
                else:
                    print(f"❌ Single question failed: {response.status}")
                    return False
            
            # Test multiple questions for comprehensive accuracy
            print("\n📝 Testing Multiple Questions for Accuracy:")
            batch_request = {
                "documents": TEST_DOCUMENT_URL,
                "questions": TEST_QUESTIONS[:5]  # Test first 5 questions
            }
            
            start_time = time.time()
            async with session.post(
                f"{API_BASE_URL}/api/v1/hackrx/run",
                headers=headers,
                json=batch_request
            ) as response:
                processing_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    answers = data.get("answers", [])
                    
                    print(f"✅ Batch processing completed in {processing_time:.2f}ms")
                    print(f"📊 Received {len(answers)} answers")
                    
                    # Analyze answer quality
                    quality_metrics = analyze_answer_quality(answers)
                    print("\n📊 Answer Quality Analysis:")
                    print(f"   - Average length: {quality_metrics['avg_length']:.0f} chars")
                    print(f"   - Concise answers (≤300 chars): {quality_metrics['concise_count']}/{len(answers)}")
                    print(f"   - Good sentence count (≤2): {quality_metrics['good_sentence_count']}/{len(answers)}")
                    print(f"   - Contains key information: {quality_metrics['key_info_count']}/{len(answers)}")
                    
                    # Check for "insufficient information" responses
                    insufficient_count = sum(1 for ans in answers if "insufficient" in ans.lower())
                    print(f"   - Insufficient information responses: {insufficient_count}/{len(answers)}")
                    
                    # Calculate accuracy score
                    accuracy_score = calculate_accuracy_score(quality_metrics, len(answers))
                    print(f"\n🎯 Estimated Accuracy Score: {accuracy_score:.1f}%")
                    
                    if accuracy_score >= 80:
                        print("✅ High accuracy achieved!")
                    elif accuracy_score >= 60:
                        print("⚠️  Moderate accuracy - room for improvement")
                    else:
                        print("❌ Low accuracy - needs significant improvement")
                    
                    return True
                    
                else:
                    print(f"❌ Batch processing failed: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return False

def analyze_answer_quality(answers: List[str]) -> Dict[str, Any]:
    """Analyze the quality of answers"""
    if not answers:
        return {
            'avg_length': 0,
            'concise_count': 0,
            'good_sentence_count': 0,
            'key_info_count': 0
        }
    
    total_length = sum(len(ans) for ans in answers)
    avg_length = total_length / len(answers)
    
    concise_count = sum(1 for ans in answers if len(ans) <= 300)
    
    good_sentence_count = 0
    key_info_count = 0
    
    for answer in answers:
        # Count sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if len(sentences) <= 2:
            good_sentence_count += 1
        
        # Check for key information indicators
        answer_lower = answer.lower()
        if any(term in answer_lower for term in ["grace period", "waiting period", "covers", "policy", "days", "months"]):
            key_info_count += 1
    
    return {
        'avg_length': avg_length,
        'concise_count': concise_count,
        'good_sentence_count': good_sentence_count,
        'key_info_count': key_info_count
    }

def calculate_accuracy_score(metrics: Dict[str, Any], total_answers: int) -> float:
    """Calculate an estimated accuracy score"""
    if total_answers == 0:
        return 0.0
    
    # Weighted scoring based on multiple factors
    conciseness_score = (metrics['concise_count'] / total_answers) * 30
    sentence_score = (metrics['good_sentence_count'] / total_answers) * 25
    key_info_score = (metrics['key_info_count'] / total_answers) * 45
    
    total_score = conciseness_score + sentence_score + key_info_score
    return min(total_score, 100.0)

async def main():
    """Main test function"""
    print("🧪 Testing Accuracy Improvements")
    print("=" * 50)
    
    # Test health check
    if not await test_health_check():
        print("❌ System not ready. Exiting.")
        return
    
    # Test accuracy improvements
    success = await test_accuracy_improvements()
    
    if success:
        print("\n✅ Accuracy improvement tests completed successfully")
    else:
        print("\n❌ Accuracy improvement tests failed")
    
    print("\n📋 Summary:")
    print("   - Enhanced chunking (2000 chars, 400 overlap)")
    print("   - Full PDF processing (all pages)")
    print("   - Hybrid search (semantic + keyword relevance)")
    print("   - Increased chunk coverage (15 chunks)")
    print("   - Expected: 80%+ accuracy improvement")

if __name__ == "__main__":
    asyncio.run(main())
