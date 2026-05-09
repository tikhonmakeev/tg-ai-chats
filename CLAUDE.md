# CLAUDE.md

## Project overview

Telegram Business Bot that auto-replies to messages using LLM (OpenAI or Anthropic) with RAG-based context from PostgreSQL + FAISS.

## Tech stack

- Python 3.14, aiogram 3.x, asyncpg, openai SDK, anthropic SDK, faiss-cpu
- PostgreSQL for persistence, FAISS for vector search
- Docker Compose for deployment (bot + postgres)

## Key files

- `bot.py` — entrypoint, creates Bot/Dispatcher, registers startup/shutdown hooks, starts polling
- `handlers.py` — aiogram Router with `business_connection`, `business_message` handlers. Filters out owner messages to prevent loops. Falls back to `bot.get_business_connection()` for unknown connections
- `ai.py` — multi-provider LLM: dispatches to OpenAI or Anthropic based on `LLM_PROVIDER`. Separate embedding client (always OpenAI-compatible). Lazy client singletons
- `rag.py` — `FAISSManager` class (one index per conversation), `build_context()` returns `(system_prompt, messages)` tuple. Recent N messages + top-K from FAISS
- `db.py` — asyncpg pool, `connections` and `messages` tables with auto-migration via CREATE IF NOT EXISTS
- `config.py` — frozen dataclass `Settings` loaded from `.env`. Builds DATABASE_URL from individual DB_* vars if DATABASE_URL not set

## Important patterns

- `build_context()` returns `(str, list[dict])` not a flat message list — system prompt is separate for Anthropic compatibility
- `get_chat_response(system, messages)` takes system prompt as first arg
- Owner filtering: `message.from_user.id == connection.user_id` → skip (prevents infinite loop)
- FAISS indices are saved to disk on every add and on shutdown
- Embeddings provider can differ from LLM provider (e.g. Anthropic for chat + OpenAI for embeddings)

## Running

```bash
docker compose up --build     # with Docker
python bot.py                 # locally (needs PostgreSQL + .env)
```

## Testing

No test suite yet. Manual testing: connect bot to Telegram Business, send messages from another account, verify responses and DB records.
