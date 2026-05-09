from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import BusinessConnection, Message

import ai
import db
import rag

logger = logging.getLogger(__name__)

router = Router(name="business")

MAX_TG_MESSAGE_LENGTH = 4096


@router.business_connection()
async def on_business_connection(event: BusinessConnection) -> None:
    status = "connected" if event.is_enabled else "disconnected"
    logger.info(
        "Business %s: user_id=%d, connection_id=%s, can_reply=%s",
        status,
        event.user.id,
        event.id,
        event.can_reply,
    )
    await db.upsert_connection(
        connection_id=event.id,
        user_id=event.user.id,
        can_reply=event.can_reply,
        is_enabled=event.is_enabled,
    )


async def _resolve_connection(
    bot: Bot, connection_id: str
) -> dict | None:
    conn = await db.get_connection(connection_id)
    if conn:
        return conn

    try:
        bc = await bot.get_business_connection(connection_id)
    except Exception as e:
        logger.error("Failed to fetch business connection %s: %s", connection_id, e)
        return None

    await db.upsert_connection(
        connection_id=bc.id,
        user_id=bc.user.id,
        can_reply=bc.can_reply,
        is_enabled=not bc.is_enabled if hasattr(bc, "is_enabled") else True,
    )
    return await db.get_connection(connection_id)


@router.business_message(F.text)
async def on_business_text_message(message: Message) -> None:
    conn_id = message.business_connection_id
    if not conn_id:
        return

    conn = await _resolve_connection(message.bot, conn_id)
    if not conn:
        logger.warning("Unknown business connection: %s", conn_id)
        return

    # Skip messages from the account owner (prevents response loop)
    if message.from_user and message.from_user.id == conn["user_id"]:
        return

    if not conn["can_reply"]:
        return

    chat_id = message.chat.id
    text = message.text

    # Save user message + embed
    msg_id = await db.save_message(
        connection_id=conn_id,
        chat_id=chat_id,
        tg_message_id=message.message_id,
        from_user_id=message.from_user.id if message.from_user else 0,
        role="user",
        content=text,
    )
    await _embed_message(conn_id, chat_id, msg_id, text)

    # Typing indicator
    try:
        await message.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING,
            business_connection_id=conn_id,
        )
    except Exception:
        pass

    # Build context and get LLM response
    system, context = await rag.build_context(conn_id, chat_id, text)
    reply_text = await ai.get_chat_response(system, context)

    # Save assistant response + embed
    reply_msg_id = await db.save_message(
        connection_id=conn_id,
        chat_id=chat_id,
        tg_message_id=None,
        from_user_id=0,
        role="assistant",
        content=reply_text,
    )
    await _embed_message(conn_id, chat_id, reply_msg_id, reply_text)

    # Send reply (split if too long)
    for chunk in _split_text(reply_text, MAX_TG_MESSAGE_LENGTH):
        await message.answer(chunk)


@router.business_message()
async def on_business_non_text(message: Message) -> None:
    conn_id = message.business_connection_id
    if not conn_id:
        return

    conn = await _resolve_connection(message.bot, conn_id)
    if not conn or not conn["can_reply"]:
        return

    if message.from_user and message.from_user.id == conn["user_id"]:
        return

    await message.answer("I can only process text messages at the moment.")


@router.message(CommandStart(deep_link=True))
async def on_start_deep_link(message: Message) -> None:
    args = message.text.split(maxsplit=1)[1] if message.text and " " in message.text else ""
    if args.startswith("bizChat"):
        chat_id = args.removeprefix("bizChat")
        await message.answer(
            f"Managing business chat {chat_id}.\nSend /help for available commands."
        )
    else:
        await message.answer("Bot is running. Connect it via Telegram Business settings.")


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer("Bot is running. Connect it via Telegram Business settings.")


async def _embed_message(
    connection_id: str, chat_id: int, message_id: int, text: str
) -> None:
    try:
        embedding = await ai.get_embedding(text)
        rag.faiss_manager.add(connection_id, chat_id, message_id, embedding)
        await db.mark_embedding(message_id)
    except Exception as e:
        logger.warning("Failed to embed message %d: %s", message_id, e)


def _split_text(text: str, max_length: int) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to split at last newline within limit
        split_pos = text.rfind("\n", 0, max_length)
        if split_pos == -1:
            # Fall back to last space
            split_pos = text.rfind(" ", 0, max_length)
        if split_pos == -1:
            # Hard split
            split_pos = max_length

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks
