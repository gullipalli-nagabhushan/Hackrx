# performance_monitor.py - Performance Monitoring and Optimization
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
import psutil
import threading

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.lock = threading.Lock()
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        with self.lock:
            self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        with self.lock:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.metrics[operation].append(duration)
                del self.start_times[operation]
                return duration
            return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        with self.lock:
            if operation in self.metrics and self.metrics[operation]:
                return sum(self.metrics[operation]) / len(self.metrics[operation])
            return 0.0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_free": disk.free
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation: str):
    """Decorator to monitor performance of async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = performance_monitor.end_timer(operation)
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"Slow operation {operation}: {duration:.2f}s")
        return wrapper
    return decorator
