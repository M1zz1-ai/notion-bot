"""Morning digest: today's plan on a TIME axis, sent to Telegram at a fixed hour.

Two deliberate departures from the first version:

1. **Time, not status.** A plan is answered by "when", so tasks are laid out by
   slot (with their project and Definition of Done), and only what is already
   Done collapses into a trailing line. Status grouping told Bogdan what state
   his day was in; it never told him what to do at 15:00.

2. **A missing plan is loud.** The old empty-day message was
   "No tasks scheduled for today. 🎉" — bit-for-bit identical whether Bogdan had
   earned a rest day or the evening planner had died. That is the worst failure
   mode a scheduler can have: it congratulates you for its own outage. The
   digest now takes ``planner_ran`` (derived from the Mac -> VPS brief) and says
   so, loudly, when the plan was never assembled.

``format_digest`` stays pure and unit-tested; the send path pulls the data and is
wrapped in ``core.errors`` resilience so a failed digest never kills the bot.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import html
import logging
from typing import Any, Protocol

from core import notion
from core.errors import run_resilient

from .brief import BRIEF_KEY, brief_for
from .timeslots import hhmm, parse_slot

logger = logging.getLogger(__name__)

__all__ = ["BRIEF_KEY", "format_digest", "planner_ran", "collect_tasks", "send_digest"]

# Cap on per-task DoD lookups (one notion-cli call each). A sane day is 2-5
# tasks; the cap only bites if the backlog cleanup has not run yet.
DOD_FETCH_LIMIT = 12

DONE = "Done"
IN_PROGRESS = "In progress"

PLANNER_MISSING = (
    "🚨 <b>THE PLANNER DID NOT RUN.</b>\n"
    "No evening brief for {day} — this list is NOT an agreed plan, "
    "just whatever happens to be sitting in Notion."
)
PLANNER_MISSING_EMPTY = (
    "🚨 <b>THE PLANNER DID NOT RUN.</b>\n"
    "No evening brief for {day}, so there is no plan for today. "
    "This is NOT a day off, this is a failure — check the day-close routine."
)


class _TgLike(Protocol):
    async def send_text(self, text: str, chat_id: int | None = ..., **kw: Any) -> Any: ...


class _StateLike(Protocol):
    def last_json(self, *parts: str, default: Any = ...) -> Any: ...


def _daystamp(day: dt.date) -> str:
    """``26.07.2026`` — the form Bogdan reads a date in, not ISO."""
    return day.strftime("%d.%m.%Y")


def _title(task: dict[str, Any]) -> str:
    """The task's title, escaped for Telegram HTML.

    Project, type and effort used to be appended here. They are metadata about a
    task, not a reason to do it at 15:00, and at eight tasks a day the brackets
    were most of the message. They live in Notion, which is one tap away.
    """
    return html.escape(task.get("title") or "Untitled", quote=False)


def format_digest(
    tasks: list[dict[str, Any]],
    *,
    planner_ran: bool | None = None,
    tz: dt.tzinfo = dt.timezone.utc,
    today: dt.date | None = None,
) -> str:
    """Format today's tasks as an HTML Telegram digest on a time axis.

    Args:
        tasks: flat notion-cli task dicts, optionally carrying a ``dod``.
        planner_ran: True = the evening brief arrived, False = it did not (the
            digest shouts), None = not checked (the digest stays quiet about it).
        tz: zone used to render slot times and to name "today".
        today: the day being reported (defaults to today in ``tz``).
    """
    day = today or dt.datetime.now(tz).date()
    head: list[str] = []
    if planner_ran is False:
        head.append((PLANNER_MISSING_EMPTY if not tasks else PLANNER_MISSING).format(day=day))
        head.append("")

    if not tasks:
        tail = "No tasks scheduled for today."
        if planner_ran is not False:
            tail += " Empty on purpose — a rest day. 🎉"
        return "\n".join([*head, f"📅 <b><u>{_daystamp(day)} Morning Digest</u></b>", "", tail])

    timed: list[tuple[dt.datetime, dict[str, Any]]] = []
    untimed: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    for task in tasks:
        if (task.get("status") or "") == DONE:
            done.append(task)
            continue
        slot = parse_slot(task.get("date"), tz)
        if slot is None:
            untimed.append(task)
        else:
            timed.append((slot, task))

    lines = [f"📅 <b><u>{_daystamp(day)} Morning Digest</u></b>"]
    # One task per stanza, separated by a blank line. A dense list of eight
    # items reads as a wall and gets skimmed; this is the whole point of the
    # rewrite, so the spacing is deliberate and not decoration.
    for slot, task in sorted(timed, key=lambda pair: pair[0]):
        mark = "⚡" if (task.get("status") or "") == IN_PROGRESS else "🎯"
        lines += ["", f"{mark} <b><i>{hhmm(slot, tz)}</i></b> <i>{_title(task)}</i>"]
    if untimed:
        lines += ["", "📅 <b><u>No time set</u></b>"]
        for task in untimed:
            lines.append(f"🎯 <i>{_title(task)}</i>")
    if done:
        # <pre> renders as a copyable block in Telegram, which is what makes the
        # finished list something you can paste into a standup instead of retype.
        body = "\n".join(f" • {_title(task)}" for task in done)
        lines += ["", f"<pre>✅ Done ({len(done)})\n{body}</pre>"]
    lines += ["", f"<b><u>Total: {len(tasks)} task(s)</u></b>"]
    return "\n".join([*head, *lines])


def planner_ran(state: _StateLike | None, day: dt.date) -> bool | None:
    """Did the evening planner produce a brief for ``day``?

    None when there is no state to ask — the digest must not accuse a planner it
    never checked on. False is a claim, and it is made only after looking.

    Day matching lives in :func:`notion.brief.brief_for` so the digest and
    the brief runner can never disagree about which day a brief belongs to.
    """
    if state is None:
        return None
    return brief_for(state, day) is not None


def collect_tasks(limit: int = DOD_FETCH_LIMIT) -> list[dict[str, Any]]:
    """Today's tasks, enriched with each open task's DoD (best effort).

    A DoD lives in the page BODY (there is no such property), so it costs one
    ``notion-cli tasks get`` per task. A task whose lookup fails is rendered
    without its DoD rather than sinking the whole digest.
    """
    tasks = notion.list_tasks(today=True) or []
    budget = limit
    for task in tasks:
        if budget <= 0 or (task.get("status") or "") == DONE or not task.get("id"):
            continue
        budget -= 1
        try:
            full = notion.get_task(task["id"])
        except Exception as exc:  # noqa: BLE001 - best effort, never fatal
            logger.warning("dod lookup failed for %s: %s", task["id"], exc)
            continue
        if isinstance(full, dict) and full.get("dod"):
            task["dod"] = full["dod"]
    return tasks


async def send_digest(
    telegram: _TgLike,
    chat_id: int | None = None,
    *,
    state: _StateLike | None = None,
    tz: dt.tzinfo = dt.timezone.utc,
    alerter: Any | None = None,
) -> None:
    """Pull today's tasks via notion-cli, format and send the digest.

    ``state`` is the redis handle used to look for the evening brief; without it
    the digest cannot tell "planned empty" from "planner never ran" and says
    nothing either way.

    Wrapped in ``run_resilient``: a notion-cli/Telegram failure is logged and
    (if no explicit alerter) surfaced via the same Telegram client, never raised.
    """

    async def work() -> None:
        day = dt.datetime.now(tz).date()
        tasks = await asyncio.to_thread(collect_tasks)
        text = format_digest(tasks, planner_ran=planner_ran(state, day), tz=tz, today=day)
        await telegram.send_text(text, chat_id=chat_id)

    await run_resilient(work, alerter=alerter or telegram, label="daily digest")
