"""Live end-to-end smoke for the notion bot, gated on real credentials.

Drives the bot's OWN modules (tools + a wired core agent) against the REAL
notion-cli, OpenAI and Telegram. Guard-railed and clearly labelled:
  * Reads tasks only (a "what's on today" agent turn) — no task is created,
    updated or archived, so the smoke never mutates Notion.
  * Sends the morning digest to the owner chat.
  * Never starts the long-poll loop.

Gating mirrors gmail-bot-py: when ``TELEGRAM_BOT_TOKEN_NOTION`` or
``OPENAI_API_KEY`` are absent it SKIPS with a clear message and exits 0 — it
never invents creds. (notion-cli auth is separate and already configured.)

Run:
  uv run python -m notion.live_smoke
"""

from __future__ import annotations

import asyncio
import logging

import openai

from core import config, tg
from core import openai_agent as core_agent
from core.errors import ConfigError

from . import digest, tools

logger = logging.getLogger("notion_bot.live_smoke")

# Keys whose absence means "skip, don't fail".
GATING_KEYS = ["TELEGRAM_BOT_TOKEN_NOTION", "OPENAI_API_KEY"]
# Keys needed to actually run the smoke once gated through.
SMOKE_KEYS = ["TELEGRAM_BOT_TOKEN_NOTION", "OPENAI_API_KEY", "TELEGRAM_CHAT_ID"]

READ_PROMPT = "List my tasks for today. Do not create, update or delete anything."


def _present(key: str) -> bool:
    try:
        return bool(config.load([key], env_path=config.MASTER_ENV_PATH).get(key))
    except ConfigError:
        return False


def _gate() -> config.Config | None:
    """Return a loaded Config if gating creds are present, else None (skip)."""
    try:
        return config.load(SMOKE_KEYS, env_path=config.MASTER_ENV_PATH)
    except ConfigError as exc:
        missing = [k for k in GATING_KEYS if not _present(k)]
        if missing:
            print(
                f"SKIP — live smoke needs {', '.join(GATING_KEYS)} in "
                f"~/.config/m1zz1/.env (missing: {', '.join(missing)}). "
                "No real creds present; nothing to do."
            )
            return None
        print(f"Config error: {exc}")
        return None


async def _run(cfg: config.Config) -> int:
    """Real E2E: a read-only agent turn + a digest, both sent to the owner chat."""
    chat_id = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    telegram = tg.TelegramClient.from_token(cfg.require("TELEGRAM_BOT_TOKEN_NOTION"), chat_id)
    client = openai.OpenAI(api_key=cfg.require("OPENAI_API_KEY"))
    agent = core_agent.OpenAIAgent(
        client, system=tools.build_system(), model=tools.NOTION_MODEL, keep_history=True
    )
    tools.register_tools(agent)

    failures = 0
    try:
        await telegram.send_text("🧪 <b>notion-bot LIVE E2E</b> — read-only agent turn + digest…")

        print("[..] step 1 — agent reads today's tasks (find_tasks tool)", flush=True)
        reply = agent.run(READ_PROMPT)
        ok1 = bool(reply.strip())
        print(f"[{'PASS' if ok1 else 'FAIL'}] agent replied ({len(reply)} chars)", flush=True)
        failures += 0 if ok1 else 1
        if ok1:
            await telegram.send_text(f"<b>Agent:</b>\n{reply}")

        print("[..] step 2 — send morning digest", flush=True)
        await digest.send_digest(telegram, chat_id=chat_id)
        print("[PASS] digest sent to owner", flush=True)
    finally:
        await telegram.close()
    print(f"\n{failures} failure(s).", flush=True)
    return failures


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _gate()
    if cfg is None:
        return 0  # skipped or config-printed; not a hard failure for CI
    return asyncio.run(_run(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
