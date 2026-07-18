"""notion.tools: agent-tool wiring over core.notion + system prompt build.

core.notion functions are monkeypatched (no real CLI/Notion call). We assert the
tools call core.notion correctly and return JSON-serializable results, and that
build_system injects the current date.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from notion import tools


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
        "habits_today",
        "check_habit",
        "habit_stats",
    }


def test_registered_tools_are_documented() -> None:
    agent = _RecordingAgent()
    tools.register_tools(agent)  # type: ignore[arg-type]
    for fn in agent.tools.values():
        assert (fn.__doc__ or "").strip(), f"{fn.__name__} needs a docstring for the agent"


# ---- tool behaviour (core.notion mocked) -------------------------------


def _wire(monkeypatch) -> tuple[_RecordingAgent, dict[str, Any]]:
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
    assert "Trading" in sys and "Gym" in sys
    assert "Low" in sys and "High" in sys


def test_build_system_documents_habits() -> None:
    sys = tools.build_system(today=date(2026, 6, 19))
    # Habits are a SEPARATE database with their own tools — the model must not
    # confuse them with tasks.
    assert "HABIT TRACKER" in sys or "Habits" in sys
    # A few slugs and a Russian mapping hint must be present so the model maps
    # "холодная вода" -> cold etc.
    assert "cold" in sys and "training" in sys and "wake-up" in sys
    assert "холодная вода" in sys


def test_build_system_rebuilds_per_day() -> None:
    # "today"/"tomorrow" track the supplied date so a long-running bot stays
    # correct across midnight (the bot rebuilds the prompt per turn).
    s1 = tools.build_system(today=date(2026, 6, 19))
    s2 = tools.build_system(today=date(2026, 6, 20))
    assert "2026-06-20" in s2 and "2026-06-21" in s2
    assert s1 != s2
