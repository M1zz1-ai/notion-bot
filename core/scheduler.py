"""Reusable async interval loop — the cron pattern for crypto/report bots.

Run a coroutine every N seconds, each cycle
wrapped in core.errors resilience so a single failure is logged/alerted and the
loop keeps running. ``asyncio.CancelledError`` stops it cleanly (for shutdown).
"""

from __future__ import annotations

import asyncio
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
    if not run_immediately:
        await asyncio.sleep(interval_seconds)
    while max_cycles is None or cycles < max_cycles:
        await run_resilient(work, alerter=alerter, label=label)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        await asyncio.sleep(interval_seconds)
    logger.info("interval loop %s stopped after %s cycles", label, cycles)
