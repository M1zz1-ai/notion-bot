"""Notion-bot handlers: a per-chat LLM agent with Notion CRUD tools + memory.

A Telegram text OR voice message drives an LLM agent that has Notion task tools
(create/find/update/complete/archive) and a per-chat conversation buffer; voice
is transcribed first via :mod:`core.stt`.

Each chat gets its OWN ``core.openai_agent.OpenAIAgent`` instance (built lazily
by an injected factory and cached), giving per-chat memory isolation. Every turn
runs inside ``core.errors.run_resilient`` so an OpenAI/STT/notion-cli failure
pings the user and is logged, but never kills the long-poll process.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from core.errors import run_resilient
from core.stt import transcribe

logger = logging.getLogger(__name__)

WELCOME = (
    "<b>📋 Notion task bot</b>\n\n"
    "Tell me what to do with your Daily Task Tracker — by text or voice.\n"
    "Examples:\n"
    "  • <i>add Gym tomorrow, effort High</i>\n"
    "  • <i>what's on for today?</i>\n"
    "  • <i>mark the trading review done</i>\n\n"
    "I create, find, update, complete and archive tasks."
)


class _TgLike(Protocol):
    async def send_text(self, text: str, chat_id: int | None = ..., **kw: Any) -> Any: ...


class _AgentLike(Protocol):
    def run(self, prompt: str) -> str: ...


class NotionBot:
    """Holds the Telegram client and a per-chat agent cache.

    Args:
        telegram: a ``core.tg.TelegramClient`` (or compatible).
        agent_factory: zero-arg callable returning a fresh ``core.openai_agent.OpenAIAgent``
            (with Notion tools registered and ``keep_history=True``). One agent
            is built and cached per chat id, giving each chat its own memory.
        owner_chat_id: chat id used for failure alerts.
    """

    def __init__(
        self,
        telegram: _TgLike,
        agent_factory: Callable[[], _AgentLike],
        owner_chat_id: int,
    ) -> None:
        self._tg = telegram
        self._agent_factory = agent_factory
        self._owner = owner_chat_id
        self._agents: dict[int, _AgentLike] = {}

    def _agent_for(self, chat_id: int) -> _AgentLike:
        """Return (building once) the chat's own agent — its conversation buffer."""
        agent = self._agents.get(chat_id)
        if agent is None:
            agent = self._agent_factory()
            self._agents[chat_id] = agent
        return agent

    # ---- handlers -------------------------------------------------------

    async def on_start(self, chat_id: int) -> None:
        """Render the welcome message."""
        await self._tg.send_text(WELCOME, chat_id=chat_id)

    async def on_text(self, chat_id: int, text: str) -> None:
        """Run the chat's agent on the message and reply with its answer."""

        async def work() -> None:
            reply = self._agent_for(chat_id).run(text)
            if reply:
                await self._tg.send_text(reply, chat_id=chat_id)

        await run_resilient(work, alerter=self._alerter(chat_id), label="notion agent")

    async def on_voice(self, chat_id: int, audio_bytes: bytes) -> None:
        """Transcribe the voice message, then run the agent on the transcript."""

        async def work() -> None:
            transcript = await transcribe(audio_bytes)
            if not transcript.strip():
                await self._tg.send_text("🎤 Could not transcribe that.", chat_id=chat_id)
                return
            reply = self._agent_for(chat_id).run(transcript)
            if reply:
                await self._tg.send_text(reply, chat_id=chat_id)

        await run_resilient(work, alerter=self._alerter(chat_id), label="notion voice")

    # ---- helpers --------------------------------------------------------

    def _alerter(self, chat_id: int) -> Any:
        """Adapt the tg client to the Alerter protocol, pinning this chat_id."""
        tg_client = self._tg
        chat_id_outer = chat_id

        class _Alerter:
            async def send_text(self, text: str, chat_id: int | None = None) -> Any:
                return await tg_client.send_text(text, chat_id=chat_id_outer)

        return _Alerter()
