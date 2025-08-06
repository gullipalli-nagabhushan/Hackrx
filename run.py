#!/usr/bin/env python3
# run.py - Application Runner
import uvicorn
from config import settings

if __name__ == "__main__":
    host = settings.APP_HOST
    port = settings.APP_PORT

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Set to False in production
        log_level="info"
    )
