# RAG Backend — Document Ingestion & Conversational RAG APIs

A production-ready FastAPI backend with two REST APIs:
1. **Document Ingestion API** — upload, chunk, embed, and store documents
2. **Conversational RAG API** — multi-turn Q&A with Redis chat memory and interview booking

---

## Architecture

```
app/
├── api/v1/endpoints/
│   ├── ingestion.py        # Document upload & processing
│   └── conversation.py     # Conversational RAG + booking
├── core/
│   ├── config.py           # Settings (env-driven)
│   └── logging.py          # Structured logger
├── db/
│   ├── qdrant.py           # Qdrant vector store client
│   ├── redis.py            # Redis chat memory client
│   └── sql.py              # SQLAlchemy async engine + session
├── models/
│   └── booking.py          # SQLAlchemy ORM models
├── schemas/
│   ├── ingestion.py        # Pydantic I/O schemas
│   └── conversation.py     # Pydantic I/O schemas
├── services/
│   ├── chunking.py         # Fixed-size & semantic chunking
│   ├── embedding.py        # Local Ollama embedding service matrix
│   ├── ingestion.py        # Ingestion orchestrator
│   ├── retrieval.py        # Custom retriever (no LangChain chain)
│   ├── llm.py              # Ollama model completion service
│   ├── memory.py           # Redis-backed chat memory
│   ├── booking.py          # Booking extraction + persistence
│   └── conversation.py     # RAG orchestrator
└── utils/
    ├── pdf.py              # PDF text extraction
    └── text.py             # Text cleaning helpers
```

## Tech Stack

| Concern | Choice |
|---|---|
| Framework | FastAPI + Uvicorn |
| Vector DB | Qdrant |
| Chat Memory | Redis |
| Relational DB | PostgreSQL (async via SQLAlchemy 2) |
| Embeddings | Ollama — `nomic-embed-text` (768 dimensions) |
| LLM | Ollama — `llama3.2` |
| PDF parsing | PyMuPDF (`fitz`) |
| Migrations | Alembic |
| Testing Suite | Pytest |

## Chunking Strategies

| Strategy | Description |
|---|---|
| `fixed` | Split by token count with configurable overlap |
| `semantic` | Sentence-boundary aware split; groups sentences until a cosine-similarity drop signals a topic boundary |

## API Reference

### Document Ingestion

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ingest/upload` | Upload + process a document |
| `GET` | `/api/v1/ingest/documents` | List metadata of all processed documents |
| `DELETE` | `/api/v1/ingest/documents/{doc_id}` | Delete document + vectors |

### Conversational RAG

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat/message` | Send a message (RAG-powered) |
| `GET` | `/api/v1/chat/history/{session_id}` | Retrieve chat history |
| `DELETE` | `/api/v1/chat/history/{session_id}` | Clear chat history |
| `GET` | `/api/v1/chat/bookings` | List all bookings |
| `GET` | `/api/v1/chat/bookings/{booking_id}` | Get single booking |

## Quick Start

```bash
# 1. Copy and fill env vars
cp .env.example .env

# 2. Start infrastructure
docker-compose up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run DB migrations
alembic upgrade head

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

## Environment Variables

See `.env.example` for all required variables.