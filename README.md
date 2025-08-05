# LLM-Powered Query-Retrieval System Backend



## 🚀 Quick Start

1. **Setup Environment Variables**
   ```bash
   # Edit .env with your API keys
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
        -H "Authorization: Bearer 6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0" \
        -H "Content-Type: application/json" \
        -d '{
          "documents": "https://example.com/policy.pdf",
          "questions": ["What is the grace period for premium payment?"]
        }'
   ```

## 🏗️ Tech Stack

- **FastAPI**: High-performance async web framework
- **Pinecone**: Vector database for semantic search  
- **GPT-4**: Large language model for query processing
- **PostgreSQL**: Primary database with pgvector extension
- **Docker**: Containerization and deployment

## 📡 API Endpoint

### POST /api/v1/hackrx/run

**Headers:**
- `Authorization: Bearer 6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0`
- `Content-Type: application/json`

**Request:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": ["Question 1?", "Question 2?"]
}
```

**Response:**
```json
{
    "answers": ["Answer 1", "Answer 2"]
}
```

## 🔧 System Architecture

1. **Document Processing** - Downloads and extracts text from PDF/DOCX
2. **Vector Storage** - Stores embeddings in Pinecone vector database  
3. **Query Processing** - Uses GPT-4 for intelligent answer generation
4. **Database Management** - PostgreSQL for metadata and analytics

## 📁 File Structure

```
llm-query-retrieval-system/
├── main.py                 # FastAPI application
├── document_processor.py   # Document processing logic
├── vector_store.py         # Pinecone vector database
├── query_engine.py         # GPT-4 query processing
├── database.py            # PostgreSQL management
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service deployment
├── .env                  # Environment variables
└── README.md             # This file
```

## 🔒 Security Features

- Bearer token authentication as specified
- Input validation and sanitization
- Comprehensive error handling
- Secure environment variable management

## ⚡ Performance Features  

- Async processing for all I/O operations
- Database connection pooling
- Efficient embedding generation with batching
- Semantic document chunking with overlap
- Production-ready logging and monitoring

## 🐳 Deployment

The system is containerized and ready for production deployment:

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```
