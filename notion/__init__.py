"""Notion task bot: Telegram text/voice -> Claude agent with Notion CRUD tools.

Phase consumer of the shared ``core``. Replicates the n8n "Notion · Task Tracker
TG Bot v2" (cGGA6bLJWJnqcKug): a conversational Claude agent over Bogdan's Daily
Task Tracker (create/find/update/complete/archive tasks) with per-chat memory and
voice input, plus a daily morning digest. Notion auth is owned by the already
configured ``notion-cli`` (via ``core.notion``) — no Notion key needed here.
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
