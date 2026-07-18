"""notion.bot — handlers, per-chat agent memory, voice path, resilience.

Telegram, the LLM agent and STT are all mocked. No network, no real bot,
no real notion-cli (the agent's tools are never invoked by the fake agent)."""

from __future__ import annotations

from typing import Any

import pytest

from notion import bot as notion_bot


class _FakeTg:
    def __init__(self) -> None:
        self.texts: list[dict[str, Any]] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append({"text": text, "chat_id": chat_id, **kw})
        return []


class _FakeAgent:
    """Stands in for a per-chat OpenAIAgent. Records prompts; replies fixed."""

    def __init__(self, reply: str = "done ✅") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _make_bot(reply: str = "done ✅") -> tuple[notion_bot.NotionBot, _FakeTg, list[_FakeAgent]]:
    fake_tg = _FakeTg()
    created: list[_FakeAgent] = []

    def factory() -> _FakeAgent:
        a = _FakeAgent(reply)
        created.append(a)
        return a

    bot = notion_bot.NotionBot(
        telegram=fake_tg,  # type: ignore[arg-type]
        agent_factory=factory,
        owner_chat_id=42,
    )
    return bot, fake_tg, created


# ---- /start ------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_start_sends_welcome() -> None:
    bot, fake_tg, _ = _make_bot()
    await bot.on_start(chat_id=7)
    assert fake_tg.texts, "expected a welcome message"


# ---- text -> agent -----------------------------------------------------


@pytest.mark.asyncio
async def test_text_runs_agent_and_replies() -> None:
    bot, fake_tg, created = _make_bot("created Gym ✅")
    await bot.on_text(chat_id=7, text="add task Gym tomorrow")
    assert created and created[0].prompts == ["add task Gym tomorrow"]
    assert any("created Gym" in t["text"] for t in fake_tg.texts)


@pytest.mark.asyncio
async def test_per_chat_memory_isolated() -> None:
    """Each chat gets its own Agent instance (its own conversation buffer)."""
    bot, _, created = _make_bot()
    await bot.on_text(chat_id=7, text="hi")
    await bot.on_text(chat_id=7, text="again")
    await bot.on_text(chat_id=99, text="other chat")
    # chat 7 reused one agent across both turns; chat 99 got a separate one.
    assert len(created) == 2
    assert created[0].prompts == ["hi", "again"]
    assert created[1].prompts == ["other chat"]


# ---- voice -> stt -> agent ---------------------------------------------


@pytest.mark.asyncio
async def test_voice_transcribes_then_runs_agent(monkeypatch) -> None:
    bot, fake_tg, created = _make_bot("ok ✅")

    async def fake_transcribe(audio: Any, **kw: Any) -> str:
        return "что у меня сегодня"

    monkeypatch.setattr(notion_bot, "transcribe", fake_transcribe)
    await bot.on_voice(chat_id=7, audio_bytes=b"\x00ogg")
    assert created and created[0].prompts == ["что у меня сегодня"]
    assert any("ok" in t["text"] for t in fake_tg.texts)


@pytest.mark.asyncio
async def test_voice_failure_is_resilient(monkeypatch) -> None:
    bot, fake_tg, _ = _make_bot()

    async def boom(audio: Any, **kw: Any) -> str:
        raise RuntimeError("whisper down")

    monkeypatch.setattr(notion_bot, "transcribe", boom)
    await bot.on_voice(chat_id=7, audio_bytes=b"x")  # must not raise
    assert any("fail" in t["text"].lower() or "⚠️" in t["text"] for t in fake_tg.texts)


# ---- agent failure is resilient ----------------------------------------


@pytest.mark.asyncio
async def test_agent_failure_pings_user_not_crash() -> None:
    bot, fake_tg, _ = _make_bot()

    def factory_boom() -> Any:
        class _Boom:
            def run(self, prompt: str) -> str:
                raise RuntimeError("openai 529")

        return _Boom()

    bot._agent_factory = factory_boom  # type: ignore[attr-defined]
    await bot.on_text(chat_id=7, text="add Gym")  # must not raise
    assert any("fail" in t["text"].lower() or "⚠️" in t["text"] for t in fake_tg.texts)
