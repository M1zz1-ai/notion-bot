"""Daily morning digest: today's tasks grouped by status, sent to Telegram.

Parses ``notion-cli``'s FLAT task JSON (``{title, status, type[], effort[]}``).
Pure ``format_digest`` is unit-tested; the send path is wrapped in
``core.errors`` resilience so a failed digest never crashes the host bot process.
"""

from __future__ import annotations

from typing import Any, Protocol

from core import notion
from core.errors import run_resilient

# Status buckets in display order, with their header emoji.
STATUS_ORDER = ("Not started", "In progress", "Done")
STATUS_EMOJI = {"Not started": "⏳", "In progress": "⚡", "Done": "✅"}


class _TgLike(Protocol):
    async def send_text(self, text: str, chat_id: int | None = ..., **kw: Any) -> Any: ...


def _label(task: dict[str, Any]) -> str:
    """Render one task line: ``Title (Type | Effort)`` (label omitted if empty)."""
    title = task.get("title") or "Untitled"
    types = ", ".join(task.get("type") or [])
    effort = ", ".join(task.get("effort") or [])
    tag = " | ".join(p for p in (types, effort) if p)
    return f"{title} ({tag})" if tag else title


def format_digest(tasks: list[dict[str, Any]]) -> str:
    """Format today's tasks into an HTML Telegram digest grouped by status.

    Unknown/missing statuses fall into the "Not started" bucket so no task is
    silently dropped. Returns a friendly message when there are no tasks.
    """
    if not tasks:
        return "☀️ <b>Morning digest</b>\n\nNo tasks scheduled for today. 🎉"

    groups: dict[str, list[str]] = {status: [] for status in STATUS_ORDER}
    for task in tasks:
        status = task.get("status") or "Not started"
        bucket = status if status in groups else "Not started"
        groups[bucket].append(_label(task))

    lines = ["☀️ <b>Morning digest — today's tasks</b>", ""]
    for status in STATUS_ORDER:
        items = groups[status]
        if not items:
            continue
        lines.append(f"{STATUS_EMOJI[status]} <b>{status}</b> ({len(items)})")
        lines.extend(f"  • {item}" for item in items)
        lines.append("")
    lines.append(f"Total: {len(tasks)} task(s)")
    return "\n".join(lines)


async def send_digest(
    telegram: _TgLike,
    chat_id: int | None = None,
    *,
    alerter: Any | None = None,
) -> None:
    """Pull today's tasks via notion-cli, format and send the digest.

    Wrapped in ``run_resilient``: a notion-cli/Telegram failure is logged and
    (if no explicit alerter) surfaced via the same Telegram client, never raised.
    """

    async def work() -> None:
        tasks = notion.list_tasks(today=True)
        await telegram.send_text(format_digest(tasks or []), chat_id=chat_id)

    await run_resilient(work, alerter=alerter or telegram, label="daily digest")
