#!/usr/bin/env python3
"""
Logging configuration for the LLM-Powered Query-Retrieval System
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config import settings

def setup_logging():
    """Setup logging configuration for the entire application"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for app.log
    file_handler = RotatingFileHandler(
        'app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers for external libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)  # For embeddings
    logging.getLogger("groq").setLevel(logging.INFO)    # For LLM
    logging.getLogger("pinecone").setLevel(logging.INFO)
    
    # Test log message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")
    logger.info(f"Log level set to: {settings.LOG_LEVEL}")
    logger.info(f"Log file: {os.path.abspath('app.log')}")
    
    return logger

# Initialize logging when this module is imported
setup_logging()
