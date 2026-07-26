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


def test_add_task_passes_project_and_dod(monkeypatch):
    """The day brief commits tasks with a project link and a DoD in the body."""
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id": "new"}'), cap)
    notion.add_task("Ship it", project="M1zz1 OS", dod="It is live.", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "add",
        "Ship it",
        "--project",
        "M1zz1 OS",
        "--dod",
        "It is live.",
        "--json",
    ]


def test_update_task_passes_project_and_dod(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id":"p1"}'), cap)
    notion.update_task("p1", project="M1zz1 OS", dod="It is live.", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "update",
        "p1",
        "--project",
        "M1zz1 OS",
        "--dod",
        "It is live.",
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


def test_json_mode_error_on_stdout_reaches_the_caller(monkeypatch):
    """notion-cli reports --json failures on STDOUT and leaves stderr EMPTY.

    Reading stderr alone produced "failed (1): " with no reason — losing exactly
    the message a user can act on (which project names actually exist).
    """
    cap = {}
    payload = {"ok": False, "error": "No project titled 'M1zz OS'. Available: M1zz1 OS"}
    _patch(monkeypatch, FakeProc(stdout=json.dumps(payload), returncode=1), cap)
    with pytest.raises(NotionError) as exc:
        notion.add_task("x", project="M1zz OS", cli="ncli")
    assert "Available: M1zz1 OS" in str(exc.value)


def test_failure_with_neither_stream_still_says_something(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(returncode=1), cap)
    with pytest.raises(NotionError, match="no error message"):
        notion.get_task("p1", cli="ncli")


# ---- project tracker (read-only) ---------------------------------------


def test_list_projects_argv_and_parse(monkeypatch):
    cap = {}
    payload = [{"id": "pr1", "title": "M1zz1 OS", "progress": 0.4}]
    _patch(monkeypatch, FakeProc(stdout=json.dumps(payload)), cap)
    assert notion.list_projects(cli="ncli") == payload
    assert cap["cmd"] == ["ncli", "projects", "list", "--json"]


def test_get_project_argv(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout=json.dumps({"id": "pr1"})), cap)
    notion.get_project("pr1", cli="ncli")
    assert cap["cmd"] == ["ncli", "projects", "get", "pr1", "--json"]


def test_add_task_forwards_project_and_dod(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout=json.dumps({"id": "p1"})), cap)
    notion.add_task("Ship it", project="M1zz1 OS", dod="Tests green", cli="ncli")
    assert "--project" in cap["cmd"] and "M1zz1 OS" in cap["cmd"]
    assert "--dod" in cap["cmd"] and "Tests green" in cap["cmd"]


def test_update_task_forwards_iso_datetime_verbatim(monkeypatch):
    """The plumbing must not normalise a datetime down to a bare date."""
    cap = {}
    _patch(monkeypatch, FakeProc(stdout=json.dumps({"id": "p1"})), cap)
    notion.update_task("p1", date="2026-07-25T21:15:00+01:00", cli="ncli")
    assert "2026-07-25T21:15:00+01:00" in cap["cmd"]


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


# ── CLI version floor ─────────────────────────────────────────────────────────
# The bot shells out to notion-cli; a host running an older build silently loses
# flags (--project/--dod, ISO datetimes). These pin the startup assertion.


def _patch_version(monkeypatch, proc):
    cap = {}

    def fake_run(cmd, capture_output, text, check):
        cap["cmd"] = cmd
        return proc

    monkeypatch.setattr(subprocess, "run", fake_run)
    return cap


def test_cli_version_argv_has_no_json_flag(monkeypatch):
    # notion-cli's --version predates --json and errors on it; assert the argv.
    cap = _patch_version(monkeypatch, FakeProc(stdout="notion-cli 0.2.1\n"))
    assert notion.cli_version(cli="ncli") == "0.2.1"
    assert cap["cmd"] == ["ncli", "--version"]


def test_check_cli_version_too_old(monkeypatch):
    _patch_version(monkeypatch, FakeProc(stdout="notion-cli 0.1.0\n"))
    result = notion.check_cli_version(cli="ncli")
    assert result.ok is False
    assert result.reason == "too_old"
    assert result.found == "0.1.0"
    assert result.required == notion.MIN_CLI_VERSION
    assert "0.1.0" in result.message and notion.MIN_CLI_VERSION in result.message


def test_check_cli_version_exactly_at_floor(monkeypatch):
    _patch_version(monkeypatch, FakeProc(stdout=f"notion-cli {notion.MIN_CLI_VERSION}\n"))
    result = notion.check_cli_version(cli="ncli")
    assert result.ok is True
    assert result.reason == "ok"
    assert result.found == notion.MIN_CLI_VERSION


def test_check_cli_version_well_above_floor(monkeypatch):
    # 0.10.0 > 0.2.1 numerically but LESS as a string — the whole point of a real parse.
    _patch_version(monkeypatch, FakeProc(stdout="notion-cli 0.10.0\n"))
    result = notion.check_cli_version(cli="ncli")
    assert result.ok is True
    assert result.found == "0.10.0"


def test_check_cli_version_missing_binary(monkeypatch):
    def fake_run(cmd, capture_output, text, check):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = notion.check_cli_version(cli="/nope/ncli")
    assert result.ok is False
    assert result.reason == "missing"
    assert result.found is None
    assert "/nope/ncli" in result.message


def test_check_cli_version_unparseable_output(monkeypatch):
    _patch_version(monkeypatch, FakeProc(stdout="I am not a version\n"))
    result = notion.check_cli_version(cli="ncli")
    assert result.ok is False
    assert result.reason == "unparseable"
    assert result.found is None
    assert "I am not a version" in result.message


def test_check_cli_version_nonzero_exit(monkeypatch):
    _patch_version(monkeypatch, FakeProc(stderr="boom", returncode=2))
    result = notion.check_cli_version(cli="ncli")
    assert result.ok is False
    assert result.reason == "unavailable"
    assert "boom" in result.message


def test_check_cli_version_honours_explicit_minimum(monkeypatch):
    _patch_version(monkeypatch, FakeProc(stdout="notion-cli 0.2.1\n"))
    result = notion.check_cli_version(cli="ncli", minimum="0.9.0")
    assert result.ok is False
    assert result.required == "0.9.0"


def test_cli_version_raises_on_garbage(monkeypatch):
    _patch_version(monkeypatch, FakeProc(stdout="garbage"))
    with pytest.raises(NotionError):
        notion.cli_version(cli="ncli")


# ── Type multi-select: shape coercion + enum validation ──────────────────────
#
# A bare string is iterable, so ``for t in types`` used to walk it CHARACTER by
# character: types="English" became --type E --type n --type g ... Seven invalid
# flags, notion-cli rejected the lot, and the bot told Bogdan his (correct) type
# was going into the tool wrong and offered to drop it. Live, 2026-07-25.


def test_bare_string_type_becomes_one_flag_not_seven(monkeypatch):
    """Regression pin. The exact argv is the assertion — a count is not enough."""
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id": "new"}'), cap)
    notion.add_task("Learn words", types="English", cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "add",
        "Learn words",
        "--type",
        "English",
        "--json",
    ]
    assert "E" not in cap["cmd"]


def test_single_element_list_is_unchanged(monkeypatch):
    """The already-correct shape must produce byte-identical argv to the string."""
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id": "new"}'), cap)
    notion.add_task("Learn words", types=["English"], cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "add",
        "Learn words",
        "--type",
        "English",
        "--json",
    ]


def test_invalid_type_raises_naming_the_valid_ones(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id": "new"}'), cap)
    with pytest.raises(NotionError) as exc:
        notion.add_task("Gym", types=["Gym"], cli="ncli")
    message = str(exc.value)
    assert "'Gym'" in message
    for valid in notion.TYPE_VALUES:
        assert valid in message
    assert "cmd" not in cap, "must fail before shelling out"


def test_invalid_type_is_caught_before_the_string_is_split(monkeypatch):
    """'english' (wrong case) must name the enum, not report seven bad letters."""
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id": "new"}'), cap)
    with pytest.raises(NotionError) as exc:
        notion.add_task("Gym", types="english", cli="ncli")
    assert "'english'" in str(exc.value)
    assert "'e'" not in str(exc.value)


def test_update_task_types_repeats_flag(monkeypatch):
    """update takes the same shape and name as add — notion-cli's --type is repeatable."""
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id":"p1"}'), cap)
    notion.update_task("p1", types=["IT", "Sport"], cli="ncli")
    assert cap["cmd"] == [
        "ncli",
        "tasks",
        "update",
        "p1",
        "--type",
        "IT",
        "--type",
        "Sport",
        "--json",
    ]


def test_update_task_accepts_a_bare_string_type(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id":"p1"}'), cap)
    notion.update_task("p1", types="Trading", cli="ncli")
    assert cap["cmd"] == ["ncli", "tasks", "update", "p1", "--type", "Trading", "--json"]


def test_update_task_rejects_an_invalid_type(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeProc(stdout='{"id":"p1"}'), cap)
    with pytest.raises(NotionError):
        notion.update_task("p1", types="Gym", cli="ncli")
    assert "cmd" not in cap
