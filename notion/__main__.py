"""Entrypoint: one asyncio process — aiogram long-polling + digest + slot pinger.

Run modes:
  python -m notion              # long-polling + morning digest + slot pinger
  python -m notion --check      # validate config loading, then exit
  python -m notion --digest-once  # send one digest now and exit (cron-friendly)

The digest fires at a WALL-CLOCK hour (``DIGEST_HOUR``, default 11:00 in
``TIMEZONE``), not on an interval measured from process start. That distinction
is the whole point: the previous ``24 * 3600`` interval meant a unit restarted at
15:00 moved "morning" to 15:00 permanently, with nothing in any log to say so.
Scheduling stays in-process rather than moving to a systemd timer because the
pinger has to live here anyway (60s tick + in-process dedup fallback), and
``--digest-once`` would pay a full bot/agent build just to send one message.

Config keys (declared required; REDIS_URL/TIMEZONE/DIGEST_HOUR have defaults):
  TELEGRAM_BOT_TOKEN_NOTION, OPENAI_API_KEY, TELEGRAM_CHAT_ID, REDIS_URL,
  TIMEZONE, DIGEST_HOUR
  OPENAI_API_KEY powers both the LLM brain (core.openai_agent) and voice STT
  (core.stt / Whisper). ANTHROPIC_API_KEY is no longer used (brain moved to
  OpenAI after the direct Anthropic key ran out of credits on the VPS).

Notion auth is NOT here — it lives in the already configured ``notion-cli``
(reached via ``core.notion``). A missing key fails loud naming the key
(core.config.ConfigError); the process never silently runs without credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo

import openai
from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from core import config, notion, tg
from core import openai_agent as core_agent
from core.errors import ConfigError
from core.scheduler import run_daily_at
from core.state import RedisState

from . import digest, tools
from .bot import NotionBot
from .brief import BriefRunner
from .pinger import Pinger

logger = logging.getLogger("notion_bot")

REQUIRED_KEYS = [
    "TELEGRAM_BOT_TOKEN_NOTION",
    "OPENAI_API_KEY",
    "TELEGRAM_CHAT_ID",
    "REDIS_URL",
    "TIMEZONE",
    "DIGEST_HOUR",
]


def verify_cli() -> notion.VersionCheck:
    """Assert the notion-cli on disk is new enough, at BOOT, and say so out loud.

    The bot's Notion half is a subprocess call away, so a host left on an older
    notion-cli loses flags silently: nothing fails until a user action happens to
    need ``--project``/``--dod``, and then it reads as a bot bug rather than a
    deploy gap. Checking at startup turns that into one line, once, naming the
    version found, the version needed, and the fix.
    """
    result = notion.check_cli_version()
    if result.ok:
        logger.info("%s", result.message)
    else:
        logger.error("%s", result.message)
    return result


async def alert_if_degraded(telegram: Any, cli_check: notion.VersionCheck) -> None:
    """Push a stale/missing notion-cli to Telegram, best-effort.

    The process starts anyway (see :func:`main`): this bot is the delivery half
    of the day loop, and a unit that refuses to boot produces exactly the silence
    the day loop exists to eliminate — Bogdan would see no morning plan and no
    reason. Starting degraded puts the reason in the same channel the plan would
    have arrived in. The deploy preflight (``--check``) is where this is fatal.
    """
    if cli_check.ok:
        return
    try:
        await telegram.send_text(f"🚨 notion-bot started DEGRADED — {cli_check.message}")
    except Exception:  # noqa: BLE001 - a failed alert must not stop the boot
        logger.exception("failed to alert degraded notion-cli")


def build_bot(
    cfg: config.Config,
    *,
    on_owner_message: Callable[[], None] | None = None,
    tz: ZoneInfo | None = None,
) -> tuple[NotionBot, tg.TelegramClient]:
    """Wire the shared core into a NotionBot from a loaded config.

    The agent factory builds a fresh tool-equipped, memory-keeping Agent per
    chat (per-chat conversation isolation), rebuilding the system prompt each
    time so "today" stays correct across days.

    ``tz`` is the configured scheduling zone, threaded into the system prompt so
    the model's "today" is Bogdan's date and not the (UTC) host's.

    ``on_owner_message`` is the day-loop hook (see :class:`BriefRunner`).
    """
    chat_id = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    telegram = tg.TelegramClient.from_token(cfg.require("TELEGRAM_BOT_TOKEN_NOTION"), chat_id)
    client = openai.OpenAI(api_key=cfg.require("OPENAI_API_KEY"))

    def agent_factory() -> core_agent.OpenAIAgent:
        agent = core_agent.OpenAIAgent(
            client,
            system=tools.build_system(tz=tz),
            model=tools.NOTION_MODEL,
            keep_history=True,
        )
        tools.register_tools(agent)
        return agent

    bot = NotionBot(
        telegram, agent_factory, owner_chat_id=chat_id, on_owner_message=on_owner_message
    )
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


def build_clock(cfg: config.Config) -> tuple[ZoneInfo, int]:
    """Resolve the scheduling zone and digest hour from config (defaults apply)."""
    tzinfo = ZoneInfo(cfg.require("TIMEZONE"))
    return tzinfo, int(cfg.require("DIGEST_HOUR"))


async def run(cfg: config.Config) -> None:
    """Long-polling + the fixed-hour digest + the slot pinger + the day brief."""
    # The brief runner needs the Telegram client, and the bot needs the runner's
    # reply hook — so the hook is late-bound rather than the wiring re-ordered.
    runner: BriefRunner | None = None

    def note_reply() -> None:
        if runner is not None:
            runner.note_reply()

    cli_check = verify_cli()
    tzinfo, digest_hour = build_clock(cfg)
    bot, telegram = build_bot(cfg, on_owner_message=note_reply, tz=tzinfo)
    await alert_if_degraded(telegram, cli_check)
    dp = build_dispatcher(bot)
    owner = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    state = RedisState(cfg.require("REDIS_URL"))
    runner = BriefRunner(telegram, chat_id=owner, state=state, tz=tzinfo)

    async def digest_cycle() -> None:
        await digest.send_digest(telegram, chat_id=owner, state=state, tz=tzinfo)

    digest_task = asyncio.create_task(
        run_daily_at(
            digest_cycle,
            digest_hour,
            tz=tzinfo,
            alerter=telegram,
            label="morning digest",
        )
    )
    pinger_task = asyncio.create_task(
        Pinger(telegram, chat_id=owner, state=state, tz=tzinfo).run(alerter=telegram)
    )
    brief_task = asyncio.create_task(runner.run(alerter=telegram))
    logger.info(
        "notion-bot started; long-polling + digest at %02d:00 %s + slot pinger + day brief",
        digest_hour,
        tzinfo.key,
    )
    try:
        await dp.start_polling(telegram.bot, handle_signals=False)
    finally:
        digest_task.cancel()
        pinger_task.cancel()
        brief_task.cancel()
        await telegram.close()


async def run_digest_once(cfg: config.Config) -> None:
    """Send a single digest now and exit (for an external cron, e.g. systemd timer)."""
    cli_check = verify_cli()
    tzinfo, _ = build_clock(cfg)
    bot, telegram = build_bot(cfg, tz=tzinfo)
    owner = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    state = RedisState(cfg.require("REDIS_URL"))
    try:
        await alert_if_degraded(telegram, cli_check)
        await digest.send_digest(telegram, chat_id=owner, state=state, tz=tzinfo)
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
        # The preflight is the one place a stale notion-cli is fatal: it runs on
        # the host before the unit is enabled, so failing here blocks the deploy
        # instead of shipping a bot that cannot do half its job.
        cli_check = verify_cli()
        if not cli_check.ok:
            print(f"notion-cli check FAILED: {cli_check.message}", file=sys.stderr)
            return 2
        print(
            f"Config OK — all {len(REQUIRED_KEYS)} required keys present; "
            f"notion-cli {cli_check.found} >= {cli_check.required}."
        )
        return 0

    try:
        asyncio.run(run_digest_once(cfg) if args.digest_once else run(cfg))
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
