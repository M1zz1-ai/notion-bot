"""Notion task bot: Telegram text/voice -> LLM agent with Notion CRUD tools.

A conversational agent over a Daily Task Tracker (create/find/update/complete/
archive tasks) and a separate Habit Tracker, with per-chat memory and voice
input, plus a daily morning digest. Notion auth is owned by the external
``notion-cli`` (via :mod:`core.notion`) — no Notion API key lives in this repo.
"""

from .bot import NotionBot
from .digest import format_digest, send_digest
from .tools import AGENT_SYSTEM, build_system, register_tools

__all__ = [
    "AGENT_SYSTEM",
    "NotionBot",
    "build_system",
    "format_digest",
    "register_tools",
    "send_digest",
]
