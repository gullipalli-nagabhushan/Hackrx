import os
from dotenv import load_dotenv

# Load .env file only once at project startup
load_dotenv()

class Settings:
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = os.getenv("POSTGRES_DB","postgres")
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")


    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0")




# Global settings instance
settings = Settings()
