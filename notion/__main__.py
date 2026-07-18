"""Entrypoint: one asyncio process — aiogram long-polling + a daily digest task.

Run modes:
  python -m notion              # run the bot (long-polling) + daily digest
  python -m notion --check      # validate config loading, then exit
  python -m notion --digest-once  # send one digest now and exit (cron-friendly)

Required config keys (see .env.example):
  TELEGRAM_BOT_TOKEN_NOTION, OPENAI_API_KEY, TELEGRAM_CHAT_ID
  OPENAI_API_KEY powers both the LLM brain (core.openai_agent) and voice STT
  (core.stt / Whisper).

Notion auth is NOT here — it lives in the external ``notion-cli`` (reached via
``core.notion``). A missing key fails loud naming the key
(core.config.ConfigError); the process never silently runs without credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import openai
from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from core import config, tg
from core import openai_agent as core_agent
from core.errors import ConfigError
from core.scheduler import run_interval

from . import digest, tools
from .bot import NotionBot

logger = logging.getLogger("notion_bot")

REQUIRED_KEYS = [
    "TELEGRAM_BOT_TOKEN_NOTION",
    "OPENAI_API_KEY",
    "TELEGRAM_CHAT_ID",
]

DIGEST_INTERVAL_SECONDS = 24 * 3600  # once a day (run_immediately=False -> after first sleep)


def build_bot(cfg: config.Config) -> tuple[NotionBot, tg.TelegramClient]:
    """Wire the shared core into a NotionBot from a loaded config.

    The agent factory builds a fresh tool-equipped, memory-keeping Agent per
    chat (per-chat conversation isolation), rebuilding the system prompt each
    time so "today" stays correct across days.
    """
    chat_id = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    telegram = tg.TelegramClient.from_token(cfg.require("TELEGRAM_BOT_TOKEN_NOTION"), chat_id)
    client = openai.OpenAI(api_key=cfg.require("OPENAI_API_KEY"))

    def agent_factory() -> core_agent.OpenAIAgent:
        agent = core_agent.OpenAIAgent(
            client, system=tools.build_system(), model=tools.NOTION_MODEL, keep_history=True
        )
        tools.register_tools(agent)
        return agent

    bot = NotionBot(telegram, agent_factory, owner_chat_id=chat_id)
    return bot, telegram


def build_dispatcher(bot: NotionBot) -> Dispatcher:
    """Wire aiogram message handlers (start, voice, text) onto the NotionBot."""
    dp = Dispatcher()

    @dp.message(Command("start", "help"))
    async def _on_start(message: Message) -> None:
        await bot.on_start(message.chat.id)

    @dp.message(lambda m: m.voice is not None)
    async def _on_voice(message: Message) -> None:
        file = await message.bot.get_file(message.voice.file_id)
        buf = await message.bot.download_file(file.file_path)
        audio_bytes = buf.read() if buf is not None else b""
        await bot.on_voice(message.chat.id, audio_bytes)

    @dp.message(lambda m: bool(m.text) and not m.text.startswith("/"))
    async def _on_text(message: Message) -> None:
        await bot.on_text(message.chat.id, message.text or "")

    return dp


async def run(cfg: config.Config) -> None:
    """Build everything and run long-polling + the daily digest concurrently."""
    bot, telegram = build_bot(cfg)
    dp = build_dispatcher(bot)
    owner = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]

    async def digest_cycle() -> None:
        await digest.send_digest(telegram, chat_id=owner)

    digest_task = asyncio.create_task(
        run_interval(
            digest_cycle,
            DIGEST_INTERVAL_SECONDS,
            alerter=telegram,
            label="daily digest",
            run_immediately=False,
        )
    )
    logger.info("notion-bot started; long-polling + daily digest")
    try:
        await dp.start_polling(telegram.bot, handle_signals=False)
    finally:
        digest_task.cancel()
        await telegram.close()


async def run_digest_once(cfg: config.Config) -> None:
    """Send a single digest now and exit (for an external cron, e.g. systemd timer)."""
    bot, telegram = build_bot(cfg)
    owner = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    try:
        await digest.send_digest(telegram, chat_id=owner)
    finally:
        await telegram.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="notion")
    parser.add_argument("--check", action="store_true", help="Validate config and exit.")
    parser.add_argument("--digest-once", action="store_true", help="Send one digest and exit.")
    args = parser.parse_args()

    try:
        cfg = config.load(REQUIRED_KEYS)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(f"Config OK — all {len(REQUIRED_KEYS)} required keys present.")
        return 0

    try:
        asyncio.run(run_digest_once(cfg) if args.digest_once else run(cfg))
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
