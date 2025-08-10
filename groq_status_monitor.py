# groq_status_monitor.py - Groq Service Status Monitor and Fallback Handler
import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class GroqStatusMonitor:
    def __init__(self):
        self.status_cache = {}
        self.last_check = None
        self.check_interval = 300  # Check every 5 minutes
        self.groq_status_url = "https://api.groq.com/health"
        self.status_page_url = "https://groqstatus.com/"
        self.is_service_healthy = True
        self.last_error_time = None
        self.error_count = 0
        self.max_errors = 3  # Consider service unhealthy after 3 consecutive errors

    async def check_service_status(self) -> Dict[str, Any]:
        """Check Groq service status and return health information"""
        try:
            current_time = time.time()
            
            # Use cached status if recent enough
            if (self.last_check and 
                current_time - self.last_check < self.check_interval and 
                self.status_cache):
                return self.status_cache

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Try to check Groq's health endpoint
                try:
                    async with session.get(self.groq_status_url) as response:
                        if response.status == 200:
                            self.is_service_healthy = True
                            self.error_count = 0
                            self.last_error_time = None
                        else:
                            self.is_service_healthy = False
                            self.error_count += 1
                            self.last_error_time = current_time
                except Exception as e:
                    logger.warning(f"Could not check Groq health endpoint: {str(e)}")
                    # Don't mark as unhealthy for connection errors to health endpoint
                    pass

            # Update status cache
            status_info = {
                "is_healthy": self.is_service_healthy,
                "last_check": datetime.now().isoformat(),
                "error_count": self.error_count,
                "last_error_time": datetime.fromtimestamp(self.last_error_time).isoformat() if self.last_error_time else None,
                "status_page_url": self.status_page_url,
                "recommendation": self._get_recommendation()
            }
            
            self.status_cache = status_info
            self.last_check = current_time
            
            return status_info

        except Exception as e:
            logger.error(f"Error checking Groq service status: {str(e)}")
            return {
                "is_healthy": False,
                "last_check": datetime.now().isoformat(),
                "error": str(e),
                "status_page_url": self.status_page_url,
                "recommendation": "Unable to check service status. Please visit the status page for updates."
            }

    def _get_recommendation(self) -> str:
        """Get recommendation based on current service status"""
        if self.is_service_healthy:
            return "Service is healthy. Proceed with normal operations."
        elif self.error_count >= self.max_errors:
            return "Service appears to be experiencing issues. Consider implementing fallback mechanisms or retry later."
        else:
            return "Service may be experiencing temporary issues. Retry with exponential backoff."

    def should_retry(self) -> bool:
        """Determine if we should retry based on error patterns"""
        if self.is_service_healthy:
            return True
        
        if self.error_count >= self.max_errors:
            return False
        
        # Allow retries for temporary issues
        return True

    def get_retry_delay(self) -> float:
        """Calculate retry delay based on error count"""
        base_delay = 1.0
        max_delay = 30.0
        delay = min(base_delay * (2 ** self.error_count), max_delay)
        return delay

    async def get_fallback_response(self, question: str, context: str) -> str:
        """Generate a fallback response when Groq is unavailable"""
        try:
            # Simple keyword-based fallback response
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Check if we can find relevant information in the context
            if context.strip() and context != "No relevant information found in the documents.":
                # Extract a simple answer from context
                sentences = context.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 200:  # Reasonable sentence length
                        # Check if sentence contains keywords from question
                        question_words = [word for word in question_lower.split() if len(word) > 3]
                        if any(word in sentence.lower() for word in question_words):
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    # Return the most relevant sentence
                    return relevant_sentences[0][:300] + "."
            
            # Generic fallback response
            return "Service temporarily unavailable. Please try again in a few minutes or visit https://groqstatus.com/ for service status updates."
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return "Service temporarily unavailable. Please try again later."

# Global instance
groq_monitor = GroqStatusMonitor()

async def get_groq_status() -> Dict[str, Any]:
    """Get current Groq service status"""
    return await groq_monitor.check_service_status()

async def should_retry_groq() -> bool:
    """Check if we should retry Groq API calls"""
    await groq_monitor.check_service_status()
    return groq_monitor.should_retry()

async def get_groq_retry_delay() -> float:
    """Get recommended retry delay for Groq API calls"""
    await groq_monitor.check_service_status()
    return groq_monitor.get_retry_delay()

async def get_fallback_answer(question: str, context: str) -> str:
    """Get a fallback answer when Groq is unavailable"""
    return await groq_monitor.get_fallback_response(question, context)
