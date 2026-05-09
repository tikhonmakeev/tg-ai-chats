# tg-ai-chats

Telegram Business Bot that responds to messages on behalf of a user using LLM (OpenAI / Anthropic Claude).

Uses the [Chat Automation in Profiles](https://core.telegram.org/bots/features#bots-for-business) feature — connect the bot to your Telegram Business profile and it will automatically reply to incoming messages.

## Architecture

- **aiogram 3.x** — Telegram Bot API (polling mode)
- **Anthropic / OpenAI** — LLM for generating responses (configurable via `LLM_PROVIDER`)
- **OpenAI** — embeddings for RAG
- **PostgreSQL** — message and connection storage
- **FAISS** — vector index for semantic search (RAG)

### Message flow

1. User connects the bot via Telegram Settings > Business > Chatbots
2. Someone sends a message to the user
3. Bot receives `business_message` update
4. Bot builds context: last N messages + top-K relevant messages from history (RAG)
5. Bot sends context to LLM, gets a response
6. Bot replies on behalf of the user

## Setup

### Prerequisites

- Python 3.14+ or Docker
- PostgreSQL
- Telegram Business subscription
- API key: OpenAI or Anthropic

### 1. Create the bot

- Talk to [@BotFather](https://t.me/BotFather), create a new bot
- Go to Bot Settings > Business Mode > enable it

### 2. Configure

Create `.env`:

```bash
# Telegram
BOT_TOKEN=your-bot-token

# LLM — "openai" or "anthropic"
LLM_PROVIDER=anthropic
LLM_API_KEY=your-anthropic-key
LLM_MODEL=claude-sonnet-4-20250514

# Embeddings (OpenAI)
EMBEDDING_API_KEY=your-openai-key
EMBEDDING_MODEL=text-embedding-3-small
```

For OpenAI as LLM:

```bash
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-key
LLM_MODEL=gpt-4o
# LLM_BASE_URL=https://openrouter.ai/api/v1  # for OpenRouter
```

### 3. Run with Docker

```bash
docker compose up --build
```

PostgreSQL starts automatically. Bot connects after DB healthcheck passes.

### 3 (alt). Run locally

```bash
pip install -r requirements.txt
# Set DATABASE_URL in .env
python bot.py
```

### 4. Connect to profile

Telegram Settings > Business > Chatbots > select your bot.

## Configuration reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `BOT_TOKEN` | yes | — | Telegram bot token |
| `LLM_PROVIDER` | no | `openai` | `openai` or `anthropic` |
| `LLM_API_KEY` | yes | — | LLM API key |
| `LLM_BASE_URL` | no | provider default | Custom endpoint (OpenRouter, etc.) |
| `LLM_MODEL` | no | auto | Model name |
| `LLM_MAX_TOKENS` | no | `4096` | Max tokens in LLM response |
| `EMBEDDING_API_KEY` | no | = `LLM_API_KEY` | Embeddings API key |
| `EMBEDDING_BASE_URL` | no | OpenAI default | Embeddings endpoint |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | Embedding model |
| `SYSTEM_PROMPT` | no | generic assistant | System prompt |
| `RECENT_MESSAGES` | no | `10` | Recent messages in context |
| `RAG_TOP_K` | no | `5` | RAG results count |
| `DATABASE_URL` | no | built from DB_* | PostgreSQL connection string |
| `DB_HOST` | no | `db` | PostgreSQL host |
| `DB_PORT` | no | `5432` | PostgreSQL port |
| `DB_USER` | no | `bot` | PostgreSQL user |
| `DB_PASSWORD` | no | `bot` | PostgreSQL password |
| `DB_NAME` | no | `tg_ai_chats` | PostgreSQL database name |
| `FAISS_DATA_DIR` | no | `faiss_data` | FAISS index directory |

## Project structure

```
bot.py          — entrypoint, dispatcher, polling
handlers.py     — aiogram router, business message handlers
ai.py           — LLM + embeddings clients (multi-provider)
rag.py          — FAISS manager, context builder
db.py           — PostgreSQL via asyncpg
config.py       — settings from .env
```
