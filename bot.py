import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

import db
import rag
from config import settings
from handlers import router


async def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    logging.info("Starting business bot (model: %s)...", settings.llm_model)
    await dp.start_polling(
        bot,
        allowed_updates=[
            "business_connection",
            "business_message",
            "edited_business_message",
            "deleted_business_messages",
            "message",
        ],
    )


async def on_startup() -> None:
    await db.init_db()
    logging.info("Bot started")


async def on_shutdown() -> None:
    rag.faiss_manager.save_all()
    await db.close_db()
    logging.info("Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())
