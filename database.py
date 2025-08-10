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
        if not all([settings.POSTGRES_USER, settings.POSTGRES_PASSWORD, settings.POSTGRES_HOST, settings.POSTGRES_PORT]):
            logger.warning("Database credentials not fully configured. Database operations will be skipped.")
            return ""
        
        return f"postgresql://{settings.POSTGRES_USER}:" \
               f"{settings.POSTGRES_PASSWORD}@" \
               f"{settings.POSTGRES_HOST}:" \
               f"{settings.POSTGRES_PORT}/" \
               f"{settings.POSTGRES_DB}"

    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            if not self.connection_string:
                logger.warning("Database connection string not available. Skipping database initialization.")
                return

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

                # Perform lightweight migrations to relax overly strict column sizes
                # 1) Ensure source_url is TEXT (long URLs with query params can exceed 50 chars)
                col_defs = await connection.fetch(
                    """
                    SELECT column_name, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'documents';
                    """
                )
                col_info = {row["column_name"]: (row["data_type"], row["character_maximum_length"]) for row in col_defs}

                # Upgrade source_url to TEXT if currently varchar/limited
                if "source_url" in col_info:
                    data_type, char_len = col_info["source_url"]
                    if data_type != "text":
                        try:
                            await connection.execute(
                                "ALTER TABLE documents ALTER COLUMN source_url TYPE TEXT;"
                            )
                            logger.info("Migrated documents.source_url to TEXT")
                        except Exception as migrate_err:
                            logger.warning(f"Could not alter documents.source_url to TEXT: {str(migrate_err)}")

                # Relax document_type length to VARCHAR(255) if smaller
                if "document_type" in col_info:
                    data_type, char_len = col_info["document_type"]
                    if data_type == "character varying" and (char_len is None or char_len < 255):
                        try:
                            await connection.execute(
                                "ALTER TABLE documents ALTER COLUMN document_type TYPE VARCHAR(255);"
                            )
                            logger.info("Migrated documents.document_type to VARCHAR(255)")
                        except Exception as migrate_err:
                            logger.warning(f"Could not alter documents.document_type to VARCHAR(255): {str(migrate_err)}")

        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")

    async def document_exists(self, source_url: str) -> bool:
        """Check if document already exists in database"""
        try:
            if not self.pool:
                logger.warning("Database pool not available. Assuming document doesn't exist.")
                return False
                
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
        """Insert document metadata into database"""
        try:
            if not self.pool:
                logger.warning("Database pool not available. Skipping document insertion.")
                return
                
            async with self.pool.acquire() as connection:
                # Sanitize and cap document_type length defensively
                raw_doc_type = metadata.get('document_type')
                doc_type = None
                if isinstance(raw_doc_type, str):
                    doc_type = raw_doc_type[:255]

                await connection.execute(
                    """
                    INSERT INTO documents (source_url, document_type, total_pages, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_url) DO NOTHING;
                    """,
                    source_url,
                    doc_type,
                    metadata.get('total_pages', 0),
                    json.dumps(metadata)
                )
                logger.info(f"Document metadata inserted for {source_url}")
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")

    async def reset_database(self):
        """Delete all data and recreate tables for a clean start."""
        try:
            if not self.pool:
                logger.warning("Database pool not available. Skipping database reset.")
                return
            async with self.pool.acquire() as connection:
                await connection.execute("DROP TABLE IF EXISTS documents;")
                await self._create_tables()
                logger.info("Database reset: tables dropped and recreated")
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")

    async def delete_document(self, source_url: str):
        """Delete a single document row by URL."""
        try:
            if not self.pool:
                logger.warning("Database pool not available. Skipping document deletion.")
                return
            async with self.pool.acquire() as connection:
                await connection.execute(
                    "DELETE FROM documents WHERE source_url = $1;",
                    source_url
                )
                logger.info(f"Deleted document metadata for {source_url}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")


