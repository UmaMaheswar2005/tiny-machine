# 🐉 Hydra-RAG (tiny-machine)

A polyglot, microservice-style **Retrieval-Augmented Generation** engine. Drop documents into a folder, they are chunked, embedded and indexed in a vector database, and a chat console answers questions grounded in that data.

**Live demo:** https://tiny-machine-rho.vercel.app/

## Architecture

```
            ┌──────────────┐   chunk + embed   ┌──────────────┐
 ./data ──▶ │ Rust ingestor│ ────────────────▶ │    Qdrant    │
 (files)    │ (file watcher)│  (Jina embeddings)│ hydra_docs   │
            └──────────────┘                   └──────▲───────┘
                                                      │ top-5 search
 Chat UI ──▶ ┌──────────────┐   embed query ──────────┘
 (static)    │  Go gateway  │ ──────────────────────▶ Jina embeddings API
             │   (Fiber)    │ ──────────────────────▶ LLM via Groq (OpenAI-compatible)
             └──────────────┘
```

| Service | Folder | What it does |
|---|---|---|
| **Go gateway** | `go-gateway/` | Fiber HTTP API and static chat console. Embeds the query, searches Qdrant, builds the prompt with retrieved context, calls the LLM, keeps per-session chat history. |
| **Rust ingestor** | `rust-ingestor/` | Watches the `./data` folder (initial scan + live create/modify events), extracts text, splits it into 500-character chunks, embeds them and upserts them into Qdrant. |
| **Python inference** | `python-inference/` | Optional FastAPI service (`/embed`) serving `all-MiniLM-L6-v2` for fully local embeddings. It is not on the default retrieval path, which uses Jina embeddings. |
| **Qdrant** | (`docker-compose.yml`) | Vector store, collection `hydra_docs`, 768 dimensions, cosine distance. |

## Features

- **Live ingestion:** new or edited files in `./data` are indexed automatically. Supports `.txt`, `.md`, `.csv`, `.json` and `.pdf`.
- **Three answer modes** driven by the system prompt: direct task generation, strict answers from local context, and a general-knowledge fallback that says when no local context matched.
- **Conversation memory:** session IDs with the last 6 messages sent as context.
- **Graceful degradation:** if retrieval fails, the gateway falls back to answering directly from the LLM instead of erroring.
- **Provider-agnostic LLM:** any OpenAI-compatible chat endpoint via `LLM_BASE_URL` / `LLM_MODEL` (Groq by default).
- **Deploy-friendly:** honours `PORT`, configurable CORS (`ALLOWED_ORIGINS`), Docker image per service.

## API (Go gateway)

| Method | Route | Description |
|---|---|---|
| `POST` | `/ask` | `{ "query": "...", "session_id": "optional" }` → retrieves context, asks the LLM, returns `{ answer, session_id }`. |
| `POST` | `/search` | `{ "query": "..." }` → raw top-5 vector matches from Qdrant. |
| `GET` | `/sessions` | List active session IDs. |
| `GET` | `/sessions/:id` | Message history for a session. |
| `DELETE` | `/sessions/:id` | Delete a session. |

Sessions are held in memory, so they reset when the gateway restarts.

## Configuration

| Variable | Used by | Purpose |
|---|---|---|
| `JINA_API_KEY` | gateway, ingestor | Embeddings (`jina-embeddings-v2-base-en`) |
| `QDRANT_URL` | gateway, ingestor | Qdrant endpoint (default `http://localhost:6333`) |
| `QDRANT_API_KEY` | gateway, ingestor | Only needed for Qdrant Cloud |
| `GROQ_API_KEY` or `LLM_API_KEY` | gateway | LLM API key |
| `LLM_BASE_URL` | gateway | Chat completions URL (default: Groq's OpenAI-compatible endpoint) |
| `LLM_MODEL` | gateway | Model name (default `openai/gpt-oss-120b`) |
| `ALLOWED_ORIGINS` | gateway | CORS origins (default `*`) |
| `PORT` | gateway | Listen port (default `3000`) |

## Running locally

**With Docker Compose** (starts Qdrant, gateway, ingestor and the optional inference service):

```bash
# add JINA_API_KEY to the gateway and ingestor "environment" blocks in docker-compose.yml, then:
export LLM_API_KEY=your_groq_key
docker compose up --build
```

The console is served by the gateway at http://localhost:3000.

**Without Docker:**

```bash
# 1. Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Ingestor (put your files in ./data next to where you run it)
cd rust-ingestor && JINA_API_KEY=... cargo run --release

# 3. Gateway
cd go-gateway && JINA_API_KEY=... GROQ_API_KEY=... go run main.go
```

## Tech stack

Go (Fiber, Resty) · Rust (Tokio, notify, reqwest, pdf-extract) · Python (FastAPI, sentence-transformers) · Qdrant · Jina embeddings · Groq-hosted LLM · Docker Compose · Vercel

## Author

Built by [Uma Maheswar Reddy V](https://github.com/UmaMaheswar2005).
