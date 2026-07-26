"""Reusable async loops — the cron pattern for crypto/report bots.

Two shapes:

- :func:`run_interval` — every N seconds, measured from process start. Right for
  polling ("check again in 60s"), WRONG for anything the user experiences as a
  time of day.
- :func:`run_daily_at` — the next occurrence of a wall-clock time in a given
  timezone. A unit restarted at 15:00 still fires at 11:00, because the target
  is computed from the clock, not from when the process happened to boot.

Mirrors gmail-bot-py's poll loop: each cycle is wrapped in core.errors
resilience so a single failure is logged/alerted and the loop keeps running.
``asyncio.CancelledError`` stops it cleanly (for shutdown).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Awaitable, Callable

from .errors import Alerter, run_resilient

logger = logging.getLogger(__name__)


async def run_interval(
    work: Callable[[], Awaitable[object]],
    interval_seconds: float,
    *,
    alerter: Alerter | None = None,
    label: str = "interval",
    run_immediately: bool = True,
    max_cycles: int | None = None,
) -> None:
    """Run ``work`` every ``interval_seconds`` forever, swallowing per-cycle errors.

    Args:
        work: zero-arg coroutine factory invoked each cycle.
        interval_seconds: delay between cycle starts (sleep is after each cycle).
        alerter: optional Telegram client pinged on a failed cycle.
        label: log/alert label.
        run_immediately: run once on entry before the first sleep.
        max_cycles: stop after this many cycles (mainly for tests); None = forever.

    Stops on ``asyncio.CancelledError`` (clean shutdown).
    """
    cycles = 0
    logger.info("interval loop %s started; every %ss", label, interval_seconds)
    if not run_immediately:
        await asyncio.sleep(interval_seconds)
    while max_cycles is None or cycles < max_cycles:
        await run_resilient(work, alerter=alerter, label=label)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        await asyncio.sleep(interval_seconds)
    logger.info("interval loop %s stopped after %s cycles", label, cycles)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def next_occurrence(
    hour: int,
    minute: int,
    tz: dt.tzinfo,
    now: dt.datetime,
) -> dt.datetime:
    """Return the next ``hour:minute`` in ``tz`` STRICTLY after ``now``.

    ``now`` may carry any timezone (UTC is the usual caller); it is converted to
    ``tz`` first, so the answer is always the local wall-clock time Bogdan sees.

    The next day is computed by date arithmetic and re-attached to ``tz`` rather
    than by adding 24 hours, so a DST shift moves the fire time by an hour of
    absolute time instead of moving it off the intended wall clock. Moscow is a
    fixed +03:00 today, but the hour of the digest must not bake that in.
    """
    local = now.astimezone(tz)
    target = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= local:
        target = dt.datetime.combine(
            local.date() + dt.timedelta(days=1), dt.time(hour, minute), tzinfo=tz
        )
    return target


async def run_daily_at(
    work: Callable[[], Awaitable[object]],
    hour: int,
    minute: int = 0,
    *,
    tz: dt.tzinfo,
    alerter: Alerter | None = None,
    label: str = "daily",
    max_cycles: int | None = None,
    now_fn: Callable[[], dt.datetime] = _utc_now,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Run ``work`` once per day at ``hour:minute`` wall-clock time in ``tz``.

    Sleeps until the next occurrence, works, then recomputes from the clock — so
    a long cycle, a clock jump or a restart can shift the *next* fire time but
    never the hour it lands on.

    Args:
        work: zero-arg coroutine factory invoked each cycle.
        hour, minute: local wall-clock target.
        tz: the timezone the target is expressed in.
        alerter: optional Telegram client pinged on a failed cycle.
        label: log/alert label.
        max_cycles: stop after this many cycles (mainly for tests); None = forever.
        now_fn, sleep_fn: injectable clock/sleep for tests.

    Stops on ``asyncio.CancelledError`` (clean shutdown).
    """
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        target = next_occurrence(hour, minute, tz, now_fn())
        delay = (target - now_fn()).total_seconds()
        logger.info("%s: next run at %s (in %.0fs)", label, target.isoformat(), delay)
        await sleep_fn(max(delay, 0.0))
        await run_resilient(work, alerter=alerter, label=label)
        cycles += 1
    logger.info("daily loop %s stopped after %s cycles", label, cycles)
