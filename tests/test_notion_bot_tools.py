"""notion.tools: agent-tool wiring over core.notion + system prompt build.

core.notion functions are monkeypatched (no real CLI/Notion call). We assert the
tools call core.notion correctly and return JSON-serializable results, and that
build_system injects the current date.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from notion import tools
from core.errors import NotionError


class _RecordingAgent:
    """Captures registered tools without a real LLM client."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, fn: Any) -> Any:
        self.tools[fn.__name__] = fn
        return fn


# ---- registration ------------------------------------------------------


def test_register_tools_registers_full_crud() -> None:
    agent = _RecordingAgent()
    tools.register_tools(agent)  # type: ignore[arg-type]
    assert set(agent.tools) == {
        "create_task",
        "find_tasks",
        "update_task",
        "complete_task",
        "archive_task",
        "list_projects",
        "get_project",
        "habits_today",
        "check_habit",
        "habit_stats",
    }


def test_no_project_write_tools_are_registered() -> None:
    """Projects are read-only from chat — see the reasoning in tools.py."""
    agent = _RecordingAgent()
    tools.register_tools(agent)  # type: ignore[arg-type]
    assert not {n for n in agent.tools if "project" in n} - {"list_projects", "get_project"}


def test_registered_tools_are_documented() -> None:
    agent = _RecordingAgent()
    tools.register_tools(agent)  # type: ignore[arg-type]
    for fn in agent.tools.values():
        assert (fn.__doc__ or "").strip(), f"{fn.__name__} needs a docstring for the agent"


# ---- tool behaviour (core.notion mocked) -------------------------------


def _wire(monkeypatch, *, current_date: str = "2026-07-25") -> tuple[_RecordingAgent, dict[str, Any]]:
    """Wire the tools over a fake core.notion.

    ``current_date`` is what ``get_task`` reports as the task's existing Date —
    the value the reschedule logic reads to decide whether a time slot exists.
    """
    calls: dict[str, Any] = {}

    def _record(name: str, ret: Any):
        def fn(*a: Any, **k: Any) -> Any:
            calls[name] = (a, k)
            return ret

        return fn

    monkeypatch.setattr(tools.notion, "add_task", _record("add", {"id": "p1"}))
    monkeypatch.setattr(tools.notion, "list_tasks", _record("list", [{"id": "p1"}]))
    monkeypatch.setattr(tools.notion, "update_task", _record("update", {"id": "p1"}))
    monkeypatch.setattr(tools.notion, "complete_task", _record("complete", {"id": "p1"}))
    monkeypatch.setattr(tools.notion, "delete_task", _record("delete", None))
    monkeypatch.setattr(
        tools.notion, "get_task", _record("get", {"id": "p1", "date": current_date})
    )
    monkeypatch.setattr(
        tools.notion,
        "list_projects",
        _record("p_list", [{"id": "pr1", "title": "M1zz1 OS", "progress": 0.4}]),
    )
    monkeypatch.setattr(
        tools.notion, "get_project", _record("p_get", {"id": "pr1", "title": "M1zz1 OS"})
    )
    monkeypatch.setattr(
        tools.notion, "list_habits_today", _record("h_today", [{"cold": True}])
    )
    monkeypatch.setattr(tools.notion, "check_habit", _record("h_check", {"cold": True}))
    monkeypatch.setattr(
        tools.notion, "habit_stats", _record("h_stats", [{"habit": "cold", "pct": 80.0}])
    )
    agent = _RecordingAgent()
    tools.register_tools(agent)  # type: ignore[arg-type]
    return agent, calls


def test_create_task_forwards_fields(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["create_task"](title="Gym", date="2026-06-19", types=["Sport"], effort="High")
    assert calls["add"][1]["date"] == "2026-06-19"
    assert calls["add"][1]["types"] == ["Sport"]
    assert calls["add"][1]["effort"] == "High"
    assert out == {"id": "p1"}


def test_create_task_minimal(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["create_task"](title="Read")
    assert calls["add"][0][0] == "Read"  # positional title


def test_find_tasks_passes_filters(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["find_tasks"](today=True, status="in progress")
    assert calls["list"][1]["today"] is True
    assert calls["list"][1]["status"] == "in progress"
    assert out == [{"id": "p1"}]


def test_find_tasks_passes_date_range(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["find_tasks"](date_from="2026-07-06", date_to="2026-07-12")
    assert calls["list"][1]["date_from"] == "2026-07-06"
    assert calls["list"][1]["date_to"] == "2026-07-12"
    assert out == [{"id": "p1"}]


def test_update_task_passes_page_id_and_fields(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["update_task"](page_id="p9", status="done", title="New")
    assert calls["update"][0][0] == "p9"
    assert calls["update"][1]["status"] == "done"
    assert calls["update"][1]["title"] == "New"


# ---- the Type multi-select has ONE name and ONE shape ------------------
#
# create_task took `types: list[str]` while update_task took `type_: str`: one
# concept, two names, two shapes. The model passed the other one's shape and
# core.notion iterated the string letter by letter (see test_notion.py).


def test_update_task_forwards_types_under_the_same_name_as_create(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["update_task"](page_id="p9", types=["English"])
    assert calls["update"][1]["types"] == ["English"]


def test_both_task_tools_accept_a_bare_string_type(monkeypatch) -> None:
    """The tool layer forwards it verbatim; core.notion owns the coercion."""
    agent, calls = _wire(monkeypatch)
    agent.tools["create_task"](title="Learn words", types="English")
    agent.tools["update_task"](page_id="p9", types="English")
    assert calls["add"][1]["types"] == "English"
    assert calls["update"][1]["types"] == "English"


def test_task_tools_take_the_same_type_argument(monkeypatch) -> None:
    """Pins the symmetry itself, so the two signatures cannot drift apart again."""
    import inspect

    agent, _ = _wire(monkeypatch)
    create = inspect.signature(agent.tools["create_task"]).parameters
    update = inspect.signature(agent.tools["update_task"]).parameters
    assert "types" in create and "types" in update
    assert create["types"].annotation == update["types"].annotation


def test_type_values_are_the_boundarys_own(monkeypatch) -> None:
    """The prompt must advertise the exact enum core.notion validates against."""
    assert tools.TYPE_VALUES is tools.notion.TYPE_VALUES
    assert "English" in tools.build_system()


# ---- rescheduling must not destroy a time slot -------------------------


def test_bare_date_reschedule_keeps_an_existing_time_slot(monkeypatch) -> None:
    """"move gym to tomorrow" — only the day changes, the 21:15 slot survives.

    Losing the time drops the task out of the slot pinger: it silently stops
    reminding while the bot reports the move as done.
    """
    agent, calls = _wire(monkeypatch, current_date="2026-07-25T21:15:00+01:00")
    agent.tools["update_task"](page_id="p9", date="2026-07-26")
    assert calls["update"][1]["date"] == "2026-07-26T21:15:00+01:00"


def test_bare_date_reschedule_of_an_all_day_task_stays_all_day(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch, current_date="2026-07-25")
    agent.tools["update_task"](page_id="p9", date="2026-07-26")
    assert calls["update"][1]["date"] == "2026-07-26"


def test_time_of_day_reschedule_sets_the_new_time(monkeypatch) -> None:
    """"move gym to 5pm" — an explicit datetime is written through untouched."""
    agent, calls = _wire(monkeypatch, current_date="2026-07-25T21:15:00+01:00")
    agent.tools["update_task"](page_id="p9", date="2026-07-26T17:00:00+01:00")
    assert calls["update"][1]["date"] == "2026-07-26T17:00:00+01:00"


def test_explicit_datetime_does_not_cost_a_read(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["update_task"](page_id="p9", date="2026-07-26T17:00:00+01:00")
    assert "get" not in calls


def test_update_without_a_date_does_not_read_the_task(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["update_task"](page_id="p9", status="done")
    assert "get" not in calls
    assert calls["update"][1]["date"] is None


def test_reschedule_fails_loudly_when_the_current_slot_is_unreadable(monkeypatch) -> None:
    """A read failure must NOT fall through to the bare date.

    Writing the bare day when we cannot see the current slot is exactly the
    silent destruction this logic exists to prevent; a loud error is recoverable.
    """
    agent, _ = _wire(monkeypatch)

    def _boom(*a: Any, **k: Any) -> Any:
        raise NotionError("notion-cli tasks get failed (1): boom")

    monkeypatch.setattr(tools.notion, "get_task", _boom)
    with pytest.raises(NotionError):
        agent.tools["update_task"](page_id="p9", date="2026-07-26")


# ---- project + dod round-trip ------------------------------------------


def test_create_task_forwards_project_and_dod(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["create_task"](
        title="Ship chunker", project="M1zz1 OS", dod="Tests green on the VPS"
    )
    assert calls["add"][1]["project"] == "M1zz1 OS"
    assert calls["add"][1]["dod"] == "Tests green on the VPS"


def test_update_task_forwards_project_and_dod(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["update_task"](page_id="p9", project="M1zz1 OS", dod="Deployed")
    assert calls["update"][1]["project"] == "M1zz1 OS"
    assert calls["update"][1]["dod"] == "Deployed"


def test_unset_project_and_dod_are_not_sent(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["create_task"](title="Read")
    assert calls["add"][1]["project"] is None
    assert calls["add"][1]["dod"] is None


def test_project_resolution_failure_reaches_the_caller(monkeypatch) -> None:
    """An unresolvable project name must surface, never be guessed at."""
    agent, _ = _wire(monkeypatch)

    def _boom(*a: Any, **k: Any) -> Any:
        raise NotionError("No project titled 'M1zz OS'. Available: M1zz1 OS, Franpos")

    monkeypatch.setattr(tools.notion, "add_task", _boom)
    with pytest.raises(NotionError, match="Available: M1zz1 OS"):
        agent.tools["create_task"](title="x", project="M1zz OS")


# ---- project reads -----------------------------------------------------


def test_list_projects(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["list_projects"]()
    assert "p_list" in calls
    assert out[0]["title"] == "M1zz1 OS"


def test_get_project(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["get_project"](page_id="pr1")
    assert calls["p_get"][0][0] == "pr1"
    assert out["title"] == "M1zz1 OS"


def test_complete_task(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["complete_task"](page_id_or_title="Gym")
    assert calls["complete"][0][0] == "Gym"


def test_archive_task(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["archive_task"](page_id="p9")
    assert calls["delete"][0][0] == "p9"
    assert out  # returns a confirmation, never None (so the model sees success)


# ---- habit tool behaviour (core.notion mocked) -------------------------


def test_habits_today_forwards(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["habits_today"]()
    assert "h_today" in calls
    assert out == [{"cold": True}]


def test_check_habit_forwards_slug_and_off(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["check_habit"](habit="cold", off=True)
    assert calls["h_check"][0][0] == "cold"
    assert calls["h_check"][1]["off"] is True
    assert out == {"cold": True}


def test_check_habit_defaults_on(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    agent.tools["check_habit"](habit="training")
    assert calls["h_check"][0][0] == "training"
    assert calls["h_check"][1]["off"] is False


def test_habit_stats_forwards_days(monkeypatch) -> None:
    agent, calls = _wire(monkeypatch)
    out = agent.tools["habit_stats"](days=30)
    assert calls["h_stats"][1]["days"] == 30
    assert out == [{"habit": "cold", "pct": 80.0}]


def test_habit_tools_are_documented(monkeypatch) -> None:
    agent, _ = _wire(monkeypatch)
    for name in ("habits_today", "check_habit", "habit_stats"):
        assert (agent.tools[name].__doc__ or "").strip()


# ---- system prompt -----------------------------------------------------


def test_build_system_injects_today() -> None:
    sys = tools.build_system(today=date(2026, 6, 19))
    assert "2026-06-19" in sys
    assert "Daily Task Tracker" in sys
    # Enum values from notion-cli must appear so the model never invents options.
    assert "Trading" in sys and "Franpos" in sys
    assert "Low" in sys and "Extreme" in sys


def test_enums_match_the_live_notion_schema() -> None:
    """'Gym' shipped in this enum for a month and does not exist in Notion.

    Telling the model to emit it guarantees a 400 the model then has to explain
    away, so the tuple is pinned to the live schema (notion_cli.config).
    """
    assert tools.TYPE_VALUES == (
        "Notion",
        "IT",
        "Franpos",
        "Sport",
        "Productivity",
        "English",
        "Trading",
    )
    assert tools.EFFORT_VALUES == ("Low", "Medium", "High", "Extreme")
    assert "Gym" not in tools.build_system(today=date(2026, 6, 19))


def test_build_system_documents_both_reschedule_phrasings() -> None:
    """The two phrasings must be spelled out — this is where slots get lost."""
    sys = tools.build_system(today=date(2026, 7, 25), tz=ZoneInfo("Europe/Dublin"))
    assert "move gym to 5pm" in sys
    assert "move gym to tomorrow" in sys
    assert "slot pinger" in sys


def test_build_system_bakes_in_the_current_utc_offset() -> None:
    """Dublin is +01:00 in July and +00:00 in January; the model must not guess."""
    summer = tools.build_system(today=date(2026, 7, 25), tz=ZoneInfo("Europe/Dublin"))
    assert "+01:00" in summer


def test_build_system_documents_project_and_dod() -> None:
    sys = tools.build_system(today=date(2026, 7, 25))
    assert "project" in sys
    assert "Definition of Done" in sys
    assert "list_projects" in sys


def test_build_system_says_projects_are_read_only() -> None:
    sys = tools.build_system(today=date(2026, 7, 25))
    assert "Read-only" in sys


def test_build_system_asks_for_markdown_not_html() -> None:
    sys = tools.build_system(today=date(2026, 7, 25))
    assert "FORMATTING" in sys
    assert "write Markdown, never HTML" in sys


def test_build_system_documents_habits() -> None:
    sys = tools.build_system(today=date(2026, 6, 19))
    # Habits are a SEPARATE database with their own tools — the model must not
    # confuse them with tasks.
    assert "HABIT TRACKER" in sys or "Habits" in sys
    # A few slugs and a Russian mapping hint must be present so the model maps
    # "холодная вода" -> cold etc.
    assert "cold" in sys and "training" in sys and "wake-up" in sys
    assert "холодная вода" in sys


def test_build_system_today_follows_the_configured_zone(monkeypatch) -> None:
    """23:30 on a UTC host is already tomorrow in Dublin — the prompt must agree.

    The VPS runs UTC. A bare ``date.today()`` there told the model it was still
    yesterday for the first hour of every Dublin day, so "add this for today"
    landed on the wrong row and looked like the model misunderstanding a date.
    """
    fixed = datetime(2026, 7, 26, 23, 30, tzinfo=timezone.utc)

    class _Clock:
        @staticmethod
        def now(tz: Any = None) -> datetime:
            return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

    monkeypatch.setattr(tools, "datetime", _Clock)
    assert "Today is 2026-07-27." in tools.build_system(tz=ZoneInfo("Europe/Dublin"))
    assert "Today is 2026-07-26." in tools.build_system(tz=timezone.utc)


def test_build_system_rebuilds_per_day() -> None:
    # "today"/"tomorrow" track the supplied date so a long-running bot stays
    # correct across midnight (the bot rebuilds the prompt per turn).
    s1 = tools.build_system(today=date(2026, 6, 19))
    s2 = tools.build_system(today=date(2026, 6, 20))
    assert "2026-06-20" in s2 and "2026-06-21" in s2
    assert s1 != s2
