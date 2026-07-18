"""notion.py: builds the right ``notion-cli tasks`` argv, parses --json output,
raises on failure. subprocess.run is monkeypatched (no real CLI/Notion call)."""

import json
import subprocess

import pytest

import core.notion as notion
from core.errors import NotionError


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _patch(monkeypatch, proc, capture):
    def fake_run(cmd, capture_output, text, check):
        capture["cmd"] = cmd
        return proc

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_list_tasks_argv_and_parse(monkeypatch):
    cap = {}
    payload = [{"id": "p1", "title": "Gym"}]
    _patch(monkeypatch, FakeProc(stdout=json.dumps(payload)), cap)
    out = notion.list_tasks(today=True, cli="ncli")
    assert out == payload
    assert cap["cmd"] == ["ncli", "tasks", "list", "--today", "--json"]


def test_list_tasks_date_range_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout="[]"), cap)
    notion.list_tasks(date_from="2026-07-06", date_to="2026-07-12", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "list",
        "--date-from",
        "2026-07-06",
        "--date-to",
        "2026-07-12",
        "--json",
    ]


def test_add_task_with_types_repeats_flag(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id": "new"}'), cap)
    notion.add_task("Refactor", types=["IT", "Sport"], effort="High", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "add",
        "Refactor",
        "--type",
        "IT",
        "--type",
        "Sport",
        "--effort",
        "High",
        "--json",
    ]


def test_complete_task_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"ok": true}'), cap)
    notion.complete_task("Gym", cli="ncli")
    assert cap["cmd"] == ["ncli", "tasks", "complete", "Gym", "--json"]


def test_delete_task_passes_yes(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout="null"), cap)
    notion.delete_task("p9", cli="ncli")
    assert cap["cmd"] == ["ncli", "tasks", "delete", "p9", "--yes", "--json"]


def test_update_task_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id":"p1"}'), cap)
    notion.update_task("p1", status="done", title="New", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "update",
        "p1",
        "--title",
        "New",
        "--status",
        "done",
        "--json",
    ]


def test_nonzero_exit_raises(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stderr="boom", returncode=1), cap)
    with pytest.raises(NotionError) as exc:
        notion.get_task("p1", cli="ncli")
    assert "boom" in str(exc.value)


def test_bad_json_raises(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout="not json"), cap)
    with pytest.raises(NotionError):
        notion.list_tasks(cli="ncli")


def test_empty_stdout_returns_none(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout=""), cap)
    assert notion.get_task("p1", cli="ncli") is None


def test_missing_cli_raises(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(NotionError):
        notion.list_tasks(cli="/no/such/cli")


# ---- habits group (notion-cli habits <sub>) ----------------------------


def test_list_habits_today_argv(monkeypatch):
    cap = {}
    payload = [{"date": "2026-07-09", "cold": True}]
    _patch(monkeypatch, FakeProc(stdout=json.dumps(payload)), cap)
    out = notion.list_habits_today(cli="ncli")
    assert out == payload
    assert cap["cmd"] == ["ncli", "habits", "today", "--json"]


def test_check_habit_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"cold": true}'), cap)
    notion.check_habit("cold", cli="ncli")
    assert cap["cmd"] == ["ncli", "habits", "check", "cold", "--json"]


def test_check_habit_off_and_date_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"cold": false}'), cap)
    notion.check_habit("cold", off=True, date="2026-07-08", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "habits",
        "check",
        "cold",
        "--date",
        "2026-07-08",
        "--off",
        "--json",
    ]


def test_habit_stats_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout="[]"), cap)
    notion.habit_stats(days=30, cli="ncli")
    assert cap["cmd"] == ["ncli", "habits", "stats", "--days", "30", "--json"]


def test_habits_nonzero_exit_raises(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stderr="boom", returncode=1), cap)
    with pytest.raises(NotionError) as exc:
        notion.list_habits_today(cli="ncli")
    assert "boom" in str(exc.value)
