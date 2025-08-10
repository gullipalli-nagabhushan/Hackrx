import os
from dotenv import load_dotenv

# Load .env file only once at project startup
load_dotenv()

class Settings:
    # Database settings
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = os.getenv("POSTGRES_DB","postgres")
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

    # Vector store settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "document-retrieval-index")
    PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))

    # OpenAI settings for embeddings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Groq Cloud API settings for LLM (ChatGPT + Llama70B)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
    # Application settings
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    
    # Performance settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()  # Enable logging for debugging
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "10000"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "5"))

    # Ultra-fast settings for <20ms total response time
    ULTRA_FAST_MODE = os.getenv("ULTRA_FAST_MODE", "true").lower() == "true"
    ULTRA_FAST_CACHE_SIZE = int(os.getenv("ULTRA_FAST_CACHE_SIZE", "50000"))
    ULTRA_FAST_TIMEOUT = int(os.getenv("ULTRA_FAST_TIMEOUT", "10"))  # Increased from 1 to 10 seconds
    ULTRA_FAST_MAX_TOKENS = int(os.getenv("ULTRA_FAST_MAX_TOKENS", "200"))  # Increased for comprehensive document access
    ULTRA_FAST_TOP_K = int(os.getenv("ULTRA_FAST_TOP_K", "15"))  # Increased to 15 for maximum context coverage

    # Security
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0")

    # Optimization flags
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
    ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
    ENABLE_ULTRA_FAST_MODE = os.getenv("ENABLE_ULTRA_FAST_MODE", "true").lower() == "true"

# Global settings instance
settings = Settings()
