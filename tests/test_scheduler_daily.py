"""core.scheduler wall-clock scheduling: next_occurrence + run_daily_at.

The defect being pinned: an interval loop measured from process start silently
relocates "morning" to whatever hour the unit was restarted at. These tests
assert the fire time is a function of the WALL CLOCK, not of start time.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.scheduler import next_occurrence, run_daily_at

MSK = ZoneInfo("Europe/Moscow")


# ---- next_occurrence ---------------------------------------------------


def test_before_the_hour_fires_today() -> None:
    now = datetime(2026, 7, 26, 9, 30, tzinfo=MSK)
    assert next_occurrence(11, 0, MSK, now) == datetime(2026, 7, 26, 11, 0, tzinfo=MSK)


def test_after_the_hour_fires_tomorrow() -> None:
    now = datetime(2026, 7, 26, 15, 0, tzinfo=MSK)
    assert next_occurrence(11, 0, MSK, now) == datetime(2026, 7, 27, 11, 0, tzinfo=MSK)


def test_exactly_on_the_hour_fires_tomorrow_not_now() -> None:
    """Strictly-after, so a cycle that finishes at 11:00:00 does not re-fire."""
    now = datetime(2026, 7, 26, 11, 0, tzinfo=MSK)
    assert next_occurrence(11, 0, MSK, now) == datetime(2026, 7, 27, 11, 0, tzinfo=MSK)


def test_utc_now_is_converted_to_local_wall_clock() -> None:
    """A caller in UTC still gets 11:00 Moscow, not 11:00 UTC."""
    now = datetime(2026, 7, 26, 6, 0, tzinfo=timezone.utc)  # 09:00 MSK
    assert next_occurrence(11, 0, MSK, now) == datetime(2026, 7, 26, 11, 0, tzinfo=MSK)


def test_restart_at_any_hour_still_lands_on_eleven() -> None:
    """The regression guard: whatever hour the process starts, the target is 11:00."""
    for hour in range(24):
        now = datetime(2026, 7, 26, hour, 17, tzinfo=MSK)
        target = next_occurrence(11, 0, MSK, now)
        assert (target.hour, target.minute) == (11, 0)
        assert target > now


def test_dst_zone_keeps_wall_clock_across_the_shift() -> None:
    """Wall clock, not +24h: Europe/Berlin springs forward on 2026-03-29."""
    berlin = ZoneInfo("Europe/Berlin")
    now = datetime(2026, 3, 28, 12, 0, tzinfo=berlin)
    target = next_occurrence(11, 0, berlin, now)
    assert (target.date(), target.hour) == (datetime(2026, 3, 29).date(), 11)


# ---- run_daily_at ------------------------------------------------------


async def test_run_daily_at_sleeps_until_the_target_then_works() -> None:
    slept: list[float] = []
    fired: list[int] = []
    clock = [datetime(2026, 7, 26, 15, 0, tzinfo=MSK)]

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += timedelta(seconds=seconds)

    async def work() -> None:
        fired.append(1)

    await run_daily_at(
        work,
        11,
        tz=MSK,
        max_cycles=2,
        now_fn=lambda: clock[0],
        sleep_fn=fake_sleep,
    )

    assert fired == [1, 1]
    # 15:00 -> next 11:00 is 20h away; then a full 24h to the following 11:00.
    assert slept == [20 * 3600, 24 * 3600]


async def test_run_daily_at_survives_a_failing_cycle() -> None:
    clock = [datetime(2026, 7, 26, 10, 0, tzinfo=MSK)]
    calls: list[int] = []

    async def fake_sleep(seconds: float) -> None:
        clock[0] += timedelta(seconds=seconds)

    async def work() -> None:
        calls.append(1)
        raise RuntimeError("notion-cli down")

    await run_daily_at(
        work, 11, tz=MSK, max_cycles=2, now_fn=lambda: clock[0], sleep_fn=fake_sleep
    )
    assert len(calls) == 2  # the loop kept running after the failure
