from __future__ import annotations

import logging

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

# --- Embeddings (always OpenAI-compatible) ---

_embed_client = AsyncOpenAI(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
)


async def get_embedding(text: str) -> list[float]:
    response = await _embed_client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


# --- Chat completions (OpenAI or Anthropic) ---

async def get_chat_response(system: str, messages: list[dict[str, str]]) -> str:
    if settings.llm_provider == "anthropic":
        return await _chat_anthropic(system, messages)
    return await _chat_openai(system, messages)


async def _chat_openai(system: str, messages: list[dict[str, str]]) -> str:
    from openai import APIConnectionError, APIError, RateLimitError

    client = _get_openai_llm_client()
    full_messages = [{"role": "system", "content": system}] + messages
    try:
        completion = await client.chat.completions.create(
            model=settings.llm_model,
            messages=full_messages,
            max_tokens=settings.llm_max_tokens,
        )
        return completion.choices[0].message.content or ""
    except RateLimitError:
        logger.warning("OpenAI rate limit hit")
        return "Too many requests right now. Please try again in a moment."
    except (APIError, APIConnectionError) as e:
        logger.error("OpenAI API error: %s", e)
        return "Sorry, I'm having trouble generating a response. Please try again later."


async def _chat_anthropic(system: str, messages: list[dict[str, str]]) -> str:
    from anthropic import APIConnectionError, APIError, AsyncAnthropic, RateLimitError

    client = _get_anthropic_client()
    try:
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    except RateLimitError:
        logger.warning("Anthropic rate limit hit")
        return "Too many requests right now. Please try again in a moment."
    except (APIError, APIConnectionError) as e:
        logger.error("Anthropic API error: %s", e)
        return "Sorry, I'm having trouble generating a response. Please try again later."


# --- Lazy client singletons ---

_openai_llm_client: AsyncOpenAI | None = None
_anthropic_client = None


def _get_openai_llm_client() -> AsyncOpenAI:
    global _openai_llm_client
    if _openai_llm_client is None:
        _openai_llm_client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _openai_llm_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic

        _anthropic_client = AsyncAnthropic(api_key=settings.llm_api_key)
    return _anthropic_client
