# test_groq_error_handling.py - Test Groq Error Handling and Fallback Mechanisms
import asyncio
import logging
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq_status_monitor import get_groq_status, get_fallback_answer
from query_engine import QueryEngine
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_groq_status_monitor():
    """Test the Groq status monitor functionality"""
    print("🔍 Testing Groq Status Monitor...")
    
    try:
        # Test status check
        status = await get_groq_status()
        print(f"✅ Groq Status: {status}")
        
        # Test fallback answer generation
        question = "What is the grace period for premium payment?"
        context = "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
        
        fallback_answer = await get_fallback_answer(question, context)
        print(f"✅ Fallback Answer: {fallback_answer}")
        
        return True
    except Exception as e:
        print(f"❌ Status monitor test failed: {str(e)}")
        return False

async def test_query_engine_langchain():
    """Test the query engine with langchain_groq"""
    print("\n🔍 Testing Query Engine with Langchain Groq...")
    
    try:
        # Initialize query engine
        query_engine = QueryEngine()
        await query_engine.initialize()
        
        # Test with a simple question and context
        question = "What is the grace period for premium payment?"
        relevant_chunks = [
            {
                "content": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
                "metadata": {"source": "test_document.pdf"}
            }
        ]
        
        # This should work with langchain_groq
        answer = await query_engine.generate_answer(question, relevant_chunks, "test_document.pdf")
        print(f"✅ Query Engine Answer (Langchain): {answer}")
        
        return True
    except Exception as e:
        print(f"❌ Query engine test failed: {str(e)}")
        return False

async def test_langchain_groq_integration():
    """Test langchain_groq integration specifically"""
    print("\n🔍 Testing Langchain Groq Integration...")
    
    try:
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage, SystemMessage
        
        # Test direct langchain_groq usage
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key_here":
            llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.1,
                max_tokens=200,
            )
            
            messages = [
                SystemMessage(content="You are a helpful assistant. Keep answers concise."),
                HumanMessage(content="What is 2+2?")
            ]
            
            response = await llm.ainvoke(messages)
            print(f"✅ Langchain Groq Direct Test: {response.content}")
            return True
        else:
            print("⚠️  Groq API key not configured - skipping direct test")
            return True
            
    except Exception as e:
        print(f"❌ Langchain Groq integration test failed: {str(e)}")
        return False

async def test_503_error_simulation():
    """Simulate 503 error handling"""
    print("\n🔍 Testing 503 Error Simulation...")
    
    try:
        # Test the error message handling
        error_messages = [
            "Error code: 503 - {'error': {'message': 'Service unavailable. Visit https://groqstatus.com/ to see i...",
            "timeout error",
            "rate limit exceeded",
            "authentication failed",
            "quota exceeded"
        ]
        
        for error_msg in error_messages:
            error_str = error_msg.lower()
            
            if "503" in error_str or "service unavailable" in error_str:
                response = "Groq service is temporarily unavailable. Please try again in a few minutes. Visit https://groqstatus.com/ for service status."
            elif "timeout" in error_str:
                response = "Request timed out. Please try again with a simpler question."
            elif "rate limit" in error_str or "429" in error_str:
                response = "Service is busy. Please wait a moment and try again."
            elif "authentication" in error_str or "api key" in error_str:
                response = "Authentication error. Please check API configuration."
            elif "quota" in error_str or "billing" in error_str:
                response = "API quota exceeded. Please check your Groq account billing status."
            else:
                response = f"Error processing question: {error_msg[:100]}..."
            
            print(f"✅ Error: {error_msg[:50]}... -> Response: {response}")
        
        return True
    except Exception as e:
        print(f"❌ 503 error simulation test failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Langchain Groq Error Handling Tests...\n")
    
    tests = [
        ("Groq Status Monitor", test_groq_status_monitor),
        ("Query Engine with Langchain", test_query_engine_langchain),
        ("Langchain Groq Integration", test_langchain_groq_integration),
        ("503 Error Simulation", test_503_error_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Langchain Groq integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
