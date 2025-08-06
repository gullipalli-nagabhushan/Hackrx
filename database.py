# database.py - PostgreSQL Database Manager
import asyncio
import logging
from typing import Dict, Any, List, Optional
import asyncpg
import json
from config import settings
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.pool = None
        self.connection_string = self._build_connection_string()

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables"""
        return f"postgresql://{settings.POSTGRES_USER}:" \
               f"{settings.POSTGRES_PASSWORD}@" \
               f"{settings.POSTGRES_HOST}:" \
               f"{settings.POSTGRES_PORT}/" \
               f"{settings.POSTGRES_DB}"

    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60,
                statement_cache_size=0 
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

    async def document_exists(self, source_url: str) -> bool:
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetchval(
                    "SELECT 1 FROM documents WHERE source_url = $1 LIMIT 1;",
                    source_url
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False
    
    async def insert_document(self, source_url: str, metadata: dict):
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO documents (source_url, document_type, total_pages, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_url) DO NOTHING;
                    """,
                    source_url,
                    metadata.get('document_type'),
                    metadata.get('total_pages', 0),
                    json.dumps(metadata)
                )
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")





