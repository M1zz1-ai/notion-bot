"""notion.digest: pure time-axis formatter, the planner-never-ran guard, and
the resilient send path. core.notion is mocked; no real CLI/TG/redis."""

from __future__ import annotations

import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from notion import digest

DUBLIN = ZoneInfo("Europe/Dublin")
DAY = dt.date(2026, 7, 26)


def task(
    title: str = "A",
    date: str = "2026-07-26T15:00:00.000+01:00",
    status: str = "Not started",
    **kw: Any,
) -> dict[str, Any]:
    return {"id": title, "title": title, "date": date, "status": status, **kw}


def fmt(tasks: list[dict[str, Any]], **kw: Any) -> str:
    return digest.format_digest(tasks, tz=DUBLIN, today=DAY, **kw)


# ---- format_digest: the time axis --------------------------------------


def test_empty_tasks_says_no_tasks() -> None:
    assert "No tasks" in fmt([])


def test_timed_tasks_are_ordered_by_slot() -> None:
    msg = fmt(
        [
            task("Late", date="2026-07-26T18:00:00+01:00"),
            task("Early", date="2026-07-26T11:00:00+01:00"),
        ]
    )
    assert msg.index("11:00") < msg.index("18:00")
    assert msg.index("Early") < msg.index("Late")


def test_slot_rendered_in_local_time_not_utc() -> None:
    """A +00:00 value must show as 16:00 Dublin, not 15:00."""
    assert "16:00" in fmt([task(date="2026-07-26T15:00:00+00:00")])


def test_project_and_dod_are_shown() -> None:
    msg = fmt([task(project="M1zz1 OS", dod="45 tests green")])
    assert "M1zz1 OS" in msg
    assert "DoD: 45 tests green" in msg


def test_date_only_tasks_go_under_no_time_set() -> None:
    msg = fmt([task("Someday", date="2026-07-26")])
    assert "No time set" in msg and "Someday" in msg


def test_done_collapses_into_its_own_section() -> None:
    msg = fmt([task("Shipped", status="Done"), task("Next")])
    assert "✅" in msg and "Done" in msg and "Shipped" in msg
    assert "Total: 2" in msg


def test_in_progress_is_marked_on_the_axis() -> None:
    msg = fmt([task("Running", status="In progress")])
    assert "⚡" in msg and "15:00" in msg


def test_type_and_effort_labelled() -> None:
    msg = fmt([task("A", type=["IT"], effort=["Low"])])
    assert "IT" in msg and "Low" in msg


def test_unknown_status_is_not_dropped() -> None:
    msg = fmt([task("Weird", status="Backlog")])
    assert "Weird" in msg and "Total: 1" in msg


def test_missing_title_falls_back() -> None:
    assert "Untitled" in fmt([{"status": "Done"}])


# ---- the silent-failure guard ------------------------------------------


def test_empty_plus_planner_ran_reads_as_a_real_rest_day() -> None:
    msg = fmt([], planner_ran=True)
    assert "No tasks" in msg
    assert "🎉" in msg
    assert "DID NOT RUN" not in msg


def test_empty_plus_planner_dead_shouts_loudly() -> None:
    """The defect: an outage must not be congratulated with a party emoji."""
    msg = fmt([], planner_ran=False)
    assert "THE PLANNER DID NOT RUN" in msg
    assert "NOT a day off" in msg
    assert "🎉" not in msg


def test_planner_dead_is_announced_even_when_tasks_exist() -> None:
    """Tasks without a brief are leftovers, not an agreed plan — say so."""
    msg = fmt([task()], planner_ran=False)
    assert "THE PLANNER DID NOT RUN" in msg
    assert msg.startswith("🚨")
    assert "A" in msg


def test_unknown_planner_state_makes_no_claim() -> None:
    msg = fmt([], planner_ran=None)
    assert "DID NOT RUN" not in msg


class _FakeState:
    def __init__(self, brief: Any) -> None:
        self._brief = brief

    def last_json(self, *parts: str, default: Any = None) -> Any:
        assert parts == digest.BRIEF_KEY
        return self._brief


def test_planner_ran_true_for_todays_brief() -> None:
    """``plan_date`` is the live brief schema — the field the routine emits."""
    assert digest.planner_ran(_FakeState({"plan_date": "2026-07-26"}), DAY) is True


def test_planner_ran_false_for_a_brief_planning_another_day() -> None:
    assert digest.planner_ran(_FakeState({"plan_date": "2026-07-25"}), DAY) is False


def test_planner_ran_true_for_a_legacy_date_field() -> None:
    assert digest.planner_ran(_FakeState({"date": "2026-07-26"}), DAY) is True


def test_planner_ran_false_for_yesterdays_brief() -> None:
    """A stale brief is not evidence that today's planner ran."""
    assert digest.planner_ran(_FakeState({"date": "2026-07-25"}), DAY) is False


def test_planner_ran_false_when_no_brief() -> None:
    assert digest.planner_ran(_FakeState(None), DAY) is False


def test_planner_ran_none_without_state() -> None:
    assert digest.planner_ran(None, DAY) is None


# ---- collect_tasks: DoD enrichment -------------------------------------


def test_collect_enriches_open_tasks_with_dod(monkeypatch) -> None:
    monkeypatch.setattr(digest.notion, "list_tasks", lambda **k: [task("A"), task("B")])
    monkeypatch.setattr(digest.notion, "get_task", lambda pid: {"dod": f"dod-{pid}"})
    assert [t["dod"] for t in digest.collect_tasks()] == ["dod-A", "dod-B"]


def test_collect_skips_done_and_respects_the_budget(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        digest.notion,
        "list_tasks",
        lambda **k: [task("A", status="Done"), task("B"), task("C")],
    )
    monkeypatch.setattr(digest.notion, "get_task", lambda pid: calls.append(pid) or {"dod": "x"})
    digest.collect_tasks(limit=1)
    assert calls == ["B"]  # Done skipped, budget exhausted before C


def test_collect_survives_a_failing_dod_lookup(monkeypatch) -> None:
    def boom(pid: str) -> Any:
        raise RuntimeError("notion-cli down")

    monkeypatch.setattr(digest.notion, "list_tasks", lambda **k: [task("A")])
    monkeypatch.setattr(digest.notion, "get_task", boom)
    tasks = digest.collect_tasks()
    assert tasks and "dod" not in tasks[0]  # rendered without a DoD, not lost


# ---- send_digest (resilient) -------------------------------------------


class _FakeTg:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append(text)
        return []


async def test_send_digest_lists_today_and_sends(monkeypatch) -> None:
    monkeypatch.setattr(
        digest.notion,
        "list_tasks",
        lambda **k: [task("X", status="Done")] if k.get("today") else [],
    )
    tg = _FakeTg()
    await digest.send_digest(tg, tz=DUBLIN)
    assert tg.texts and "X" in tg.texts[0]


async def test_send_digest_reports_a_dead_planner(monkeypatch) -> None:
    monkeypatch.setattr(digest.notion, "list_tasks", lambda **k: [])
    tg = _FakeTg()
    await digest.send_digest(tg, tz=DUBLIN, state=_FakeState(None))
    assert "THE PLANNER DID NOT RUN" in tg.texts[0]


async def test_send_digest_resilient_on_failure(monkeypatch) -> None:
    def boom(**k: Any) -> Any:
        raise RuntimeError("notion-cli down")

    monkeypatch.setattr(digest.notion, "list_tasks", boom)
    tg = _FakeTg()
    # Must not raise; failure swallowed and alerted via the same tg client.
    await digest.send_digest(tg, tz=DUBLIN)
    assert any("fail" in t.lower() or "⚠️" in t for t in tg.texts)


@pytest.mark.parametrize("hour", [0, 9, 15, 23])
def test_digest_hour_is_independent_of_start_time(hour: int) -> None:
    """Companion to test_scheduler_daily: the digest's own contract is the hour."""
    from core.scheduler import next_occurrence

    now = dt.datetime(2026, 7, 26, hour, 42, tzinfo=DUBLIN)
    assert next_occurrence(11, 0, DUBLIN, now).hour == 11


@pytest.mark.parametrize("utc_hour", [0, 3, 9, 12, 15, 23])
def test_digest_hour_is_11_dublin_on_a_utc_host(utc_hour: int) -> None:
    """The VPS clock is UTC; the digest hour is Bogdan's, not the host's.

    Whatever hour the unit boots at, the next fire lands on 11:00 Dublin — which
    in July is 10:00 UTC. Computing the target from the host clock instead would
    put the digest an hour off all summer and back on time every winter, which is
    the shape of bug nobody ever reports.
    """
    from core.scheduler import next_occurrence

    now = dt.datetime(2026, 7, 26, utc_hour, 42, tzinfo=dt.timezone.utc)
    target = next_occurrence(11, 0, DUBLIN, now)
    assert target.astimezone(DUBLIN).hour == 11
    assert target.astimezone(dt.timezone.utc).hour == 10
    assert target > now
