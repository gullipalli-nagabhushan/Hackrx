# database.py - PostgreSQL Database Manager
import asyncio
import logging
from typing import Dict, Any, List, Optional
import asyncpg
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.pool = None
        self.connection_string = self._build_connection_string()

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables"""
        return f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:" \
               f"{os.getenv('POSTGRES_PASSWORD', 'password')}@" \
               f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
               f"{os.getenv('POSTGRES_PORT', '5432')}/" \
               f"{os.getenv('POSTGRES_DB', 'document_db')}"

    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Create tables
            await self._create_tables()

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            # Don't raise here to allow system to work without DB

    async def _create_tables(self):
        """Create necessary database tables"""
        try:
            async with self.pool.acquire() as connection:
                # Enable pgvector extension
                await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create documents table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        source_url TEXT UNIQUE NOT NULL,
                        document_type VARCHAR(50),
                        total_pages INTEGER,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    );
                """)

        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")


