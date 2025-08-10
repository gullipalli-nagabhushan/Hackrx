#!/usr/bin/env python3
# run.py - Application Runner
import uvicorn
from config import settings

# Import logging configuration first to ensure it's set up
import logging_config

if __name__ == "__main__":
    host = settings.APP_HOST
    port = settings.APP_PORT
    
    print(f"Starting LLM-Powered Query-Retrieval System on {host}:{port}")
    print("Logging configured - check app.log for detailed logs")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Set to False in production
        log_level="info"
    )
