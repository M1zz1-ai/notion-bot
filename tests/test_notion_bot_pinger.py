"""notion.pinger: slot pings that survive restarts, reschedules and a dead redis.

Every test drives an injected clock; nothing sleeps and nothing touches redis,
Telegram or notion-cli for real.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from notion import pinger as pinger_mod
from notion.pinger import Pinger

DUBLIN = ZoneInfo("Europe/Dublin")


class FakeTg:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append(text)
        return []


class FakeState:
    """Stands in for RedisState with a working redis."""

    def __init__(self) -> None:
        self.marks: set[str] = set()

    def seen(self, item_id: str, *, ledger: str = "seen") -> bool:
        return f"{ledger}:{item_id}" in self.marks

    def mark_seen(self, item_id: str, *, ledger: str = "seen", ttl: int = 0) -> None:
        self.marks.add(f"{ledger}:{item_id}")


class DownState:
    """RedisState's degraded contract: seen() always False, mark_seen() a no-op.

    Correct for mail dedup ("rather process twice than drop"), catastrophic for
    a ping — it would re-fire every single tick.
    """

    def seen(self, item_id: str, *, ledger: str = "seen") -> bool:
        return False

    def mark_seen(self, item_id: str, *, ledger: str = "seen", ttl: int = 0) -> None:
        return None


def task(
    page_id: str = "p1",
    title: str = "Ship the pinger",
    date: str = "2026-07-26T15:00:00",
    status: str = "Not started",
    project: str | None = "M1zz1 OS",
) -> dict[str, Any]:
    return {"id": page_id, "title": title, "date": date, "status": status, "project": project}


@pytest.fixture
def at_15_05() -> datetime:
    """Five minutes past a 15:00 Dublin slot, expressed in UTC (IST = +01:00)."""
    return datetime(2026, 7, 26, 14, 5, tzinfo=timezone.utc)


def build(tg, state, tasks, clock, **kw) -> Pinger:
    return Pinger(
        tg,
        chat_id=1,
        state=state,
        tz=DUBLIN,
        now_fn=lambda: clock[0],
        list_tasks=lambda: list(tasks),
        **kw,
    )


# ---- firing ------------------------------------------------------------


async def test_fires_inside_the_window(at_15_05) -> None:
    tg, clock = FakeTg(), [at_15_05]
    await build(tg, FakeState(), [task()], clock).tick()
    assert len(tg.texts) == 1
    assert "15:00" in tg.texts[0] and "Ship the pinger" in tg.texts[0]
    assert "M1zz1 OS" in tg.texts[0]


async def test_does_not_fire_before_the_slot(at_15_05) -> None:
    clock = [at_15_05 - timedelta(minutes=10)]  # 14:55 DUBLIN
    tg = FakeTg()
    await build(tg, FakeState(), [task()], clock).tick()
    assert tg.texts == []


async def test_does_not_fire_after_the_window(at_15_05) -> None:
    clock = [at_15_05 + timedelta(minutes=40)]  # 45 min late
    tg = FakeTg()
    p = build(tg, FakeState(), [task()], clock)
    p.mark_started()  # suppress the first-tick roll-up so we test the window alone
    await p.tick()
    assert tg.texts == []


async def test_skips_done_and_in_progress(at_15_05) -> None:
    tg, clock = FakeTg(), [at_15_05]
    tasks = [
        task("p1", "Done one", status="Done"),
        task("p2", "Running one", status="In progress"),
    ]
    await build(tg, FakeState(), tasks, clock).tick()
    assert tg.texts == []


async def test_skips_date_only_tasks(at_15_05) -> None:
    """No time component -> the morning digest owns it, not the pinger."""
    tg, clock = FakeTg(), [at_15_05]
    await build(tg, FakeState(), [task(date="2026-07-26")], clock).tick()
    assert tg.texts == []


# ---- legacy offsets ----------------------------------------------------


async def test_legacy_moscow_offset_row_fires_on_its_own_instant() -> None:
    """A historical +03:00 row is a MOMENT, not a wall clock.

    Bogdan moved to Dublin; the rows written while the tracker assumed Moscow
    are still in the database. 15:00+03:00 is 12:00 UTC = 13:00 Dublin, and that
    is when it must fire — reattaching the configured zone to it would drag it
    to 14:00 UTC and put every legacy ping two hours out.
    """
    tg = FakeTg()
    clock = [datetime(2026, 7, 26, 12, 5, tzinfo=timezone.utc)]  # 13:05 Dublin
    legacy = task(date="2026-07-26T15:00:00+03:00")
    await build(tg, FakeState(), [legacy], clock).tick()
    assert len(tg.texts) == 1
    assert "13:00" in tg.texts[0]  # rendered in Dublin, fired on the +03:00 instant


async def test_legacy_offset_row_does_not_fire_at_the_naive_hour() -> None:
    """The other half of the same contract: it must not ALSO fire at 15:00 Dublin."""
    tg = FakeTg()
    clock = [datetime(2026, 7, 26, 14, 5, tzinfo=timezone.utc)]  # 15:05 Dublin
    p = build(tg, FakeState(), [task(date="2026-07-26T15:00:00+03:00")], clock)
    p.mark_started()  # isolate the window from the stale roll-up
    await p.tick()
    assert tg.texts == []


# ---- dedup: restart, reschedule, redis ---------------------------------


async def test_second_tick_does_not_repeat(at_15_05) -> None:
    tg, clock = FakeTg(), [at_15_05]
    p = build(tg, FakeState(), [task()], clock)
    await p.tick()
    clock[0] += timedelta(minutes=1)
    await p.tick()
    assert len(tg.texts) == 1


async def test_restart_does_not_repeat(at_15_05) -> None:
    """The whole point of the redis ledger: a unit restart is not a new ping."""
    state, tg, clock = FakeState(), FakeTg(), [at_15_05]
    await build(tg, state, [task()], clock).tick()
    clock[0] += timedelta(minutes=2)
    await build(tg, state, [task()], clock).tick()  # fresh process, same ledger
    assert len(tg.texts) == 1


async def test_reschedule_rearms_the_ping(at_15_05) -> None:
    """Keyed on page_id + slot: moving a task to a new time must ping again."""
    state, tg, clock = FakeState(), FakeTg(), [at_15_05]
    await build(tg, state, [task()], clock).tick()

    clock[0] += timedelta(hours=1)  # 16:05 Dublin
    moved = task(date="2026-07-26T16:00:00")
    p = build(tg, state, [moved], clock)
    p.mark_started()
    await p.tick()
    assert len(tg.texts) == 2
    assert "16:00" in tg.texts[1]


async def test_redis_down_still_fires_at_most_once_per_process(at_15_05) -> None:
    """DownState.seen() is always False; the in-process set is what saves us."""
    tg, clock = FakeTg(), [at_15_05]
    p = build(tg, DownState(), [task()], clock)
    for _ in range(5):
        await p.tick()
        clock[0] += timedelta(minutes=1)
    assert len(tg.texts) == 1


# ---- stale slots -------------------------------------------------------


async def test_long_outage_rolls_up_instead_of_back_firing(at_15_05) -> None:
    """Three missed slots -> one summary line, not three late pings."""
    tg, clock = FakeTg(), [at_15_05 + timedelta(hours=5)]
    tasks = [
        task("p1", "Slot A", date="2026-07-26T11:00:00"),
        task("p2", "Slot B", date="2026-07-26T13:00:00"),
        task("p3", "Slot C", date="2026-07-26T15:00:00"),
    ]
    await build(tg, FakeState(), tasks, clock).tick()
    assert len(tg.texts) == 1
    assert "3" in tg.texts[0]
    assert "Slot A" in tg.texts[0] and "Slot C" in tg.texts[0]


async def test_rollup_only_on_the_first_tick(at_15_05) -> None:
    tg, clock = FakeTg(), [at_15_05 + timedelta(hours=5)]
    p = build(tg, FakeState(), [task("p1", "Slot A", date="2026-07-26T11:00:00")], clock)
    await p.tick()
    clock[0] += timedelta(minutes=1)
    await p.tick()
    assert len(tg.texts) == 1


async def test_rollup_deduped_across_restarts(at_15_05) -> None:
    state, tg = FakeState(), FakeTg()
    clock = [at_15_05 + timedelta(hours=5)]
    tasks = [task("p1", "Slot A", date="2026-07-26T11:00:00")]
    await build(tg, state, tasks, clock).tick()
    await build(tg, state, tasks, clock).tick()  # restart loop, same day
    assert len(tg.texts) == 1


async def test_rollup_ignores_slots_that_were_pinged_on_time(at_15_05) -> None:
    """A restart in the evening must not report the day's delivered pings as missed.

    The normal shape of a day: boot in the morning (nothing stale), ping at the
    slot, restart later. Only the ledger knows the ping went out — a roll-up
    built from "is it late" alone accuses the bot of missing its own work.
    """
    state, tg = FakeState(), FakeTg()
    clock = [at_15_05 - timedelta(hours=7)]  # 08:05 Dublin boot, nothing due yet
    tasks = [task("p1", "Slot A", date="2026-07-26T15:00:00")]
    await build(tg, state, tasks, clock).tick()
    assert tg.texts == []

    clock[0] = at_15_05  # 15:05 — the slot fires
    await build(tg, state, tasks, clock).tick()
    assert len(tg.texts) == 1 and "15:00" in tg.texts[0]

    clock[0] = at_15_05 + timedelta(hours=5)  # 20:05 systemd restart
    await build(tg, state, tasks, clock).tick()
    assert len(tg.texts) == 1, f"restart re-announced a delivered ping: {tg.texts[1:]}"


async def test_rollup_still_reports_a_genuinely_missed_slot(at_15_05) -> None:
    """The dedup must not silence the case the roll-up exists for."""
    state, tg = FakeState(), FakeTg()
    clock = [at_15_05 - timedelta(hours=7)]
    pinged = task("p1", "Slot A", date="2026-07-26T15:00:00")
    missed = task("p2", "Slot B", date="2026-07-26T16:00:00")
    await build(tg, state, [pinged, missed], clock).tick()

    clock[0] = at_15_05
    await build(tg, state, [pinged, missed], clock).tick()  # only Slot A is due

    clock[0] = at_15_05 + timedelta(hours=5)
    await build(tg, state, [pinged, missed], clock).tick()
    assert len(tg.texts) == 2
    assert "Slot B" in tg.texts[1] and "Slot A" not in tg.texts[1]


async def test_no_rollup_when_nothing_was_missed(at_15_05) -> None:
    tg, clock = FakeTg(), [at_15_05 - timedelta(hours=2)]
    await build(tg, FakeState(), [task()], clock).tick()
    assert tg.texts == []


# ---- polling budget ----------------------------------------------------


async def test_notion_is_polled_at_most_every_five_minutes(at_15_05) -> None:
    """60s ticks over a cache: 10 ticks must not become 10 notion-cli calls."""
    calls = [0]

    def counting_list() -> list[dict[str, Any]]:
        calls[0] += 1
        return [task()]

    clock = [at_15_05 - timedelta(hours=1)]
    p = Pinger(
        FakeTg(),
        chat_id=1,
        state=FakeState(),
        tz=DUBLIN,
        now_fn=lambda: clock[0],
        list_tasks=counting_list,
    )
    for _ in range(10):  # ticks at t = 0..9 minutes
        await p.tick()
        clock[0] += timedelta(minutes=1)
    assert calls[0] == 2  # refreshed at t=0 and t=5 only


async def test_tick_survives_a_notion_failure(at_15_05) -> None:
    def boom() -> list[dict[str, Any]]:
        raise RuntimeError("notion-cli down")

    tg = FakeTg()
    p = Pinger(
        tg,
        chat_id=1,
        state=FakeState(),
        tz=DUBLIN,
        now_fn=lambda: at_15_05,
        list_tasks=boom,
    )
    with pytest.raises(RuntimeError):
        await p.tick()  # run_interval's run_resilient is what swallows it


def test_default_list_tasks_asks_notion_for_today(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        pinger_mod.notion, "list_tasks", lambda **kw: seen.update(kw) or [{"id": "x"}]
    )
    assert pinger_mod._today_tasks() == [{"id": "x"}]
    assert seen == {"today": True}
