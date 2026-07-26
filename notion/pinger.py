"""Slot pinger: one Telegram nudge per scheduled task, at the scheduled minute.

Runs beside the morning digest inside the same process (``__main__.run``). The
digest answers "what is today"; the pinger answers "it is now" — and the two
failure modes it has to survive are the ones that look like features:

* **Restart.** The unit restarts several times a day. Firing is gated on a redis
  ledger keyed by ``page_id`` + the slot's UTC instant, so a restart replays
  nothing — while rescheduling a task to a new time correctly re-arms it,
  because the key changed.
* **Redis down.** :meth:`core.state.RedisState.seen` answers False when redis is
  unreachable ("treat as unseen"). That is right for mail dedup and wrong here:
  it would re-send the same ping every tick. An in-process set therefore sits in
  front of the ledger, degrading a redis outage to at-most-once-per-process
  instead of once-per-minute.
* **Long outage.** Coming back at 20:00 must not machine-gun the day's missed
  slots. Anything older than the fire window is summarized in ONE roll-up line
  on the first tick. The roll-up reads the SAME ledger the pings do: a slot that
  was already announced is not "missed", it is done. Without that check every
  evening restart accused the bot of missing the pings it had delivered on time
  — the one message in this bot that must never cry wolf.

The tick is 60s but Notion is polled at most every ``poll_seconds`` (default 5
min): a 60s Notion poll is ~1440 subprocess calls a day to learn nothing.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Callable
from typing import Any, Protocol

from core import notion
from core.scheduler import run_interval

from .timeslots import hhmm, parse_slot, slot_key

logger = logging.getLogger(__name__)

TICK_SECONDS = 60
DEFAULT_POLL_SECONDS = 300
DEFAULT_WINDOW = dt.timedelta(minutes=30)
PING_LEDGER = "ping"
PING_TTL = 48 * 3600

# Statuses that mean "already handled" — no nudge for them.
SKIP_STATUSES = frozenset({"Done", "In progress"})

# One candidate slot: (moment, task, ledger key). The key is built once, where
# the slot is parsed, so firing and the roll-up can never disagree about it.
_Slot = tuple[dt.datetime, dict[str, Any], str]


class _TgLike(Protocol):
    async def send_text(self, text: str, chat_id: int | None = ..., **kw: Any) -> Any: ...


class _StateLike(Protocol):
    def seen(self, item_id: str, *, ledger: str = ...) -> bool: ...
    def mark_seen(self, item_id: str, *, ledger: str = ..., ttl: int = ...) -> None: ...


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _today_tasks() -> list[dict[str, Any]]:
    """Default source: today's tasks straight from notion-cli."""
    return notion.list_tasks(today=True) or []


class Pinger:
    """Fires slot reminders for today's tasks.

    Args:
        telegram: a ``core.tg.TelegramClient`` (or compatible).
        chat_id: where reminders go.
        state: a ``core.state.RedisState`` (or compatible dedup ledger).
        tz: zone used for naive Notion values and for display.
        poll_seconds: minimum seconds between Notion polls (tick stays 60s).
        window: how late a slot may be and still fire.
        now_fn / list_tasks: injectable clock and task source (tests).
    """

    def __init__(
        self,
        telegram: _TgLike,
        *,
        chat_id: int,
        state: _StateLike,
        tz: dt.tzinfo,
        poll_seconds: int = DEFAULT_POLL_SECONDS,
        window: dt.timedelta = DEFAULT_WINDOW,
        now_fn: Callable[[], dt.datetime] = _utc_now,
        list_tasks: Callable[[], list[dict[str, Any]]] = _today_tasks,
    ) -> None:
        self._tg = telegram
        self._chat_id = chat_id
        self._state = state
        self._tz = tz
        self._poll_seconds = poll_seconds
        self._window = window
        self._now = now_fn
        self._list_tasks = list_tasks
        self._cache: list[dict[str, Any]] = []
        self._fetched_at: dt.datetime | None = None
        self._fired: set[str] = set()
        self._started = False

    # ---- dedup ---------------------------------------------------------

    def _seen(self, key: str) -> bool:
        """True if this key already fired — in this process OR per the ledger."""
        return key in self._fired or self._state.seen(key, ledger=PING_LEDGER)

    def _mark(self, key: str) -> None:
        self._fired.add(key)
        self._state.mark_seen(key, ledger=PING_LEDGER, ttl=PING_TTL)

    def mark_started(self) -> None:
        """Suppress the first-tick roll-up (used by tests and by a warm restart)."""
        self._started = True

    # ---- data ----------------------------------------------------------

    async def _tasks(self, now: dt.datetime) -> list[dict[str, Any]]:
        """Today's tasks from the cache, refreshing at most every poll_seconds."""
        fresh = (
            self._fetched_at is not None
            and (now - self._fetched_at).total_seconds() < self._poll_seconds
        )
        if not fresh:
            self._cache = await asyncio.to_thread(self._list_tasks)
            self._fetched_at = now
        return self._cache

    # ---- tick ----------------------------------------------------------

    async def tick(self) -> None:
        """One pass: fire due slots, and on the very first pass roll up stale ones.

        Raises whatever the task source raises — the caller (``run_interval``)
        owns resilience, so a notion-cli blip is logged and retried in 60s.
        """
        now = self._now()
        due: list[_Slot] = []
        stale: list[_Slot] = []

        for task in await self._tasks(now):
            if (task.get("status") or "") in SKIP_STATUSES:
                continue
            slot = parse_slot(task.get("date"), self._tz)
            if slot is None:  # date-only or unset -> the morning digest owns it
                continue
            key = f"{task.get('id', '')}:{slot_key(slot)}"
            late = now - slot
            if dt.timedelta(0) <= late < self._window:
                due.append((slot, task, key))
            elif late >= self._window and not self._seen(key):
                # Already announced (ping or earlier roll-up) -> not missed.
                stale.append((slot, task, key))

        if not self._started:
            self._started = True
            await self._send_rollup(stale)

        for slot, task, key in sorted(due, key=lambda item: item[0]):
            if self._seen(key):
                continue
            await self._tg.send_text(self._format_ping(slot, task), chat_id=self._chat_id)
            self._mark(key)
            logger.info("ping sent for %s at %s", task.get("id"), slot.isoformat())

    async def run(self, *, alerter: Any | None = None, max_cycles: int | None = None) -> None:
        """Tick every 60 seconds forever, each tick wrapped in core resilience."""
        await run_interval(
            self.tick,
            TICK_SECONDS,
            alerter=alerter or self._tg,
            label="slot pinger",
            max_cycles=max_cycles,
        )

    # ---- messages ------------------------------------------------------

    def _format_ping(self, slot: dt.datetime, task: dict[str, Any]) -> str:
        """One reminder (user-facing Telegram text)."""
        title = task.get("title") or "Untitled"
        lines = [f"⏰ <b>{hhmm(slot, self._tz)}</b> — {title}"]
        project = task.get("project")
        if project:
            lines.append(f"Project: {project}")
        return "\n".join(lines)

    async def _send_rollup(self, stale: list[_Slot]) -> None:
        """Announce missed slots ONCE, as a summary — never as late pings.

        ``stale`` arrives already filtered against the ledger, and each slot in
        it is marked afterwards: a later restart the same day reports only what
        is still genuinely unannounced, instead of replaying the whole day.
        """
        if not stale:
            return
        ordered = sorted(stale, key=lambda item: item[0])
        titles = ", ".join(task.get("title") or "Untitled" for _, task, _key in ordered)
        text = (
            f"⚠️ Slots missed today: {len(stale)} — {titles}.\n"
            "No separate reminders will be sent for them."
        )
        await self._tg.send_text(text, chat_id=self._chat_id)
        for _slot, _task, key in ordered:
            self._mark(key)
