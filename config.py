from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = os.getenv("DB_USER", "bot")
    password = os.getenv("DB_PASSWORD", "bot")
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "tg_ai_chats")
    scheme = "postgresql"
    return scheme + "://" + user + ":" + password + "@" + host + ":" + port + "/" + name


@dataclass(frozen=True)
class Settings:
    # Telegram
    bot_token: str
    # PostgreSQL
    database_url: str
    # LLM (chat completions)
    llm_provider: str  # "openai" or "anthropic"
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    llm_max_tokens: int
    # Embeddings (always OpenAI-compatible)
    embedding_api_key: str
    embedding_base_url: str | None
    embedding_model: str
    # Bot behavior
    system_prompt: str
    recent_messages: int
    rag_top_k: int
    faiss_data_dir: str


_provider = os.getenv("LLM_PROVIDER", "openai").lower()
if _provider not in ("openai", "anthropic"):
    raise ValueError(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{_provider}'")

_default_model = "claude-sonnet-4-20250514" if _provider == "anthropic" else "gpt-4o"

settings = Settings(
    bot_token=_require("BOT_TOKEN"),
    database_url=_build_database_url(),
    # LLM
    llm_provider=_provider,
    llm_api_key=_require("LLM_API_KEY"),
    llm_base_url=os.getenv("LLM_BASE_URL") or None,
    llm_model=os.getenv("LLM_MODEL", _default_model),
    llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
    # Embeddings — separate provider, defaults to OpenAI
    embedding_api_key=os.getenv("EMBEDDING_API_KEY") or _require("LLM_API_KEY"),
    embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or None,
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    # Behavior
    system_prompt=os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant responding on behalf of the user.",
    ),
    recent_messages=int(os.getenv("RECENT_MESSAGES", "10")),
    rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
    faiss_data_dir=os.getenv("FAISS_DATA_DIR", "faiss_data"),
)
