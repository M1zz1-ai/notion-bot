"""notion.digest: pure formatter over notion-cli's flat task JSON, plus the
resilient send path. core.notion.list_tasks is mocked; no real CLI/TG."""

from __future__ import annotations

from typing import Any

import pytest

from notion import digest

# ---- format_digest (pure) ----------------------------------------------


def test_empty_tasks_says_no_tasks() -> None:
    msg = digest.format_digest([])
    assert "No tasks" in msg or "no tasks" in msg


def test_groups_by_status_with_counts() -> None:
    tasks = [
        {"title": "A", "status": "Not started", "type": ["IT"], "effort": ["Low"]},
        {"title": "B", "status": "In progress", "type": [], "effort": []},
        {"title": "C", "status": "Done", "type": ["Sport"], "effort": ["High"]},
        {"title": "D", "status": "Not started", "type": [], "effort": []},
    ]
    msg = digest.format_digest(tasks)
    assert "Not started" in msg and "(2)" in msg
    assert "In progress" in msg
    assert "Done" in msg
    assert "Total: 4" in msg


def test_type_and_effort_labelled() -> None:
    tasks = [{"title": "A", "status": "Not started", "type": ["IT"], "effort": ["Low"]}]
    msg = digest.format_digest(tasks)
    assert "A" in msg
    assert "IT" in msg and "Low" in msg


def test_unknown_status_bucketed_not_started() -> None:
    tasks = [{"title": "Weird", "status": "Backlog", "type": [], "effort": []}]
    msg = digest.format_digest(tasks)
    # Falls into the Not started bucket rather than being dropped.
    assert "Weird" in msg
    assert "Total: 1" in msg


def test_missing_title_falls_back() -> None:
    tasks = [{"status": "Done"}]
    msg = digest.format_digest(tasks)
    assert "Untitled" in msg


# ---- send_digest (resilient) -------------------------------------------


class _FakeTg:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append(text)
        return []


@pytest.mark.asyncio
async def test_send_digest_lists_today_and_sends(monkeypatch) -> None:
    monkeypatch.setattr(
        digest.notion,
        "list_tasks",
        lambda **k: [{"title": "X", "status": "Done", "type": [], "effort": []}]
        if k.get("today")
        else [],
    )
    tg = _FakeTg()
    await digest.send_digest(tg)
    assert tg.texts and "X" in tg.texts[0]


@pytest.mark.asyncio
async def test_send_digest_resilient_on_failure(monkeypatch) -> None:
    def boom(**k: Any) -> Any:
        raise RuntimeError("notion-cli down")

    monkeypatch.setattr(digest.notion, "list_tasks", boom)
    tg = _FakeTg()
    # Must not raise; failure swallowed and alerted via the same tg client.
    await digest.send_digest(tg)
    assert any("fail" in t.lower() or "⚠️" in t for t in tg.texts)
