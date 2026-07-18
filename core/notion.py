"""Thin Python wrapper over an external ``notion-cli`` tool via subprocess.

``notion-cli`` owns Notion auth and the task/habit database schema. We do NOT
reimplement Notion auth here; we shell out and parse its ``--json`` output and
expose each capability as an agent-compatible callable.

The CLI path defaults to ``~/bin/notion-cli`` and is overridable per call via
the ``cli=`` argument. Each function maps to ``notion-cli <group> <subcommand>``
and returns the parsed JSON (dict/list); a non-zero exit raises NotionError with
stderr.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .errors import NotionError

logger = logging.getLogger(__name__)

DEFAULT_CLI = str(Path.home() / "bin" / "notion-cli")


def _run(cli: str, args: list[str], *, group: str = "tasks") -> Any:
    """Run ``<cli> <group> <args> --json`` and return parsed JSON.

    ``group`` selects the notion-cli command group ("tasks" or "habits").

    Raises:
        NotionError: on non-zero exit or unparseable output.
    """
    cmd = [cli, group, *args, "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise NotionError(f"notion-cli not found at {cli}") from exc
    if proc.returncode != 0:
        raise NotionError(
            f"notion-cli {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise NotionError(f"notion-cli {' '.join(args)} returned non-JSON: {exc}") from exc


def add_task(
    title: str,
    *,
    date: str | None = None,
    types: list[str] | None = None,
    effort: str | None = None,
    status: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Add a task. ``types`` maps to repeatable ``--type`` flags."""
    args = ["add", title]
    if date:
        args += ["--date", date]
    for t in types or []:
        args += ["--type", t]
    if effort:
        args += ["--effort", effort]
    if status:
        args += ["--status", status]
    return _run(cli, args)


def list_tasks(
    *,
    today: bool = False,
    date: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """List tasks, optionally filtered by today/date/status or a date range.

    ``date_from``/``date_to`` map to notion-cli's ``--date-from``/``--date-to``,
    which push a server-side Date range filter (no client-side 100-row cap).
    """
    args = ["list"]
    if today:
        args.append("--today")
    if date:
        args += ["--date", date]
    if date_from:
        args += ["--date-from", date_from]
    if date_to:
        args += ["--date-to", date_to]
    if status:
        args += ["--status", status]
    return _run(cli, args)


def get_task(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Get a single task by page id."""
    return _run(cli, ["get", page_id])


def update_task(
    page_id: str,
    *,
    title: str | None = None,
    status: str | None = None,
    date: str | None = None,
    type_: str | None = None,
    effort: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Update one or more fields of a task."""
    args = ["update", page_id]
    if title:
        args += ["--title", title]
    if status:
        args += ["--status", status]
    if date:
        args += ["--date", date]
    if type_:
        args += ["--type", type_]
    if effort:
        args += ["--effort", effort]
    return _run(cli, args)


def complete_task(page_id_or_title: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Mark a task Done (accepts page id or fuzzy title)."""
    return _run(cli, ["complete", page_id_or_title])


def delete_task(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Archive (soft-delete) a task by page id. ``--yes`` skips confirmation."""
    return _run(cli, ["delete", page_id, "--yes"])


# ── HABIT TRACKER group (notion-cli habits <sub>) ─────────────────────────────
# A SEPARATE Notion database (one row per day, a checkbox per habit) — not the
# task tracker. Slugs live in notion-cli's config.HABIT_PROPS.


def list_habits_today(*, cli: str = DEFAULT_CLI) -> Any:
    """Return today's habit row (list of flattened rows, possibly empty)."""
    return _run(cli, ["today"], group="habits")


def check_habit(
    habit: str,
    *,
    off: bool = False,
    date: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Set (or clear with ``off=True``) a habit checkbox for a day.

    ``habit`` is a slug like ``cold``/``training``/``wake-up``. ``date`` is
    YYYY-MM-DD (default: today). The day's row is created by notion-cli if it
    does not exist yet.
    """
    args = ["check", habit]
    if date:
        args += ["--date", date]
    if off:
        args.append("--off")
    return _run(cli, args, group="habits")


def habit_stats(*, days: int = 7, cli: str = DEFAULT_CLI) -> Any:
    """Return density % per habit over the last ``days`` (missing day = not done)."""
    return _run(cli, ["stats", "--days", str(days)], group="habits")
