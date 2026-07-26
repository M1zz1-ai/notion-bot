"""Notion-bot capabilities: core.notion CRUD exposed as core.agent tools.

Each capability is a small callable registered on a ``core.agent.Agent`` (the
monorepo's unification constraint), so a future unified agent can route across
every bot's tools. The tools shell out via :mod:`core.notion` (which wraps the
already-configured ``notion-cli`` — we do NOT reimplement Notion auth here).

The agent system prompt is lifted from the n8n "Notion · Task Tracker TG Bot v2"
(cGGA6bLJWJnqcKug) "AI Agent" node, with the option enums aligned to what
``notion-cli`` actually accepts (see :data:`TYPE_VALUES` / :data:`EFFORT_VALUES`;
Status: not started/in progress/done).
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, tzinfo
from typing import Any, Protocol

from core import notion

# A date with no time-of-day. Notion stores task dates as either a bare day or a
# full ISO 8601 instant, and the difference is load-bearing: only a dated-with-
# time task is visible to the slot pinger.
_BARE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# LLM brain: OpenAI (the direct Anthropic key ran out of credits on the VPS). A
# fast/cheap reasoning model handles the tool-calling task/habit CRUD loop; bump
# to gpt-5.5 via ``NOTION_MODEL`` if accuracy (e.g. week-range date math)
# regresses. ``gpt-5.4-mini`` is verified against the live /v1/models list.
NOTION_MODEL = os.getenv("NOTION_MODEL", "gpt-5.4-mini")

# notion-cli's accepted option values — the model must never invent others.
# Mirrors notion_cli.config.VALID_TYPES / VALID_EFFORTS, live-verified 2026-07-25.
# "Gym" was in this tuple for a month and has never existed in Notion: the model
# was being told to emit a value the API rejects with a 400.
#
# TYPE_VALUES is re-exported from core.notion, not restated: that module now
# ENFORCES the enum before spawning the CLI, and a prompt advertising one list
# while the boundary rejects against another is a bug with no symptom until a
# task silently fails to be created.
TYPE_VALUES = notion.TYPE_VALUES
EFFORT_VALUES = ("Low", "Medium", "High", "Extreme")
STATUS_VALUES = ("not started", "in progress", "done")

# HABIT TRACKER slugs (mirror of notion-cli config.HABIT_PROPS keys). The model
# maps a Russian habit name to one of these slugs before calling check_habit.
HABIT_SLUGS = (
    "cold",
    "coffee",
    "daily-analysis",
    "meditation",
    "training",
    "trading",
    "water",
    "wake-up",
    "course",
    "book",
    "sleep",
)

# Russian → slug hints, injected into both the system prompt and the
# check_habit tool description so the model resolves phrasing reliably.
_HABITS_HINT = (
    "холодная вода/холодный душ→cold, кофе→coffee, "
    "дневной анализ/анализ дня→daily-analysis, медитация→meditation, "
    "тренировка/зал→training, трейдинг→trading, вода→water, "
    "подъём/ранний подъём/встал рано→wake-up, курс/учёба→course, "
    "книга/чтение→book, сон→sleep"
)


class _AgentLike(Protocol):
    def tool(self, fn: Any) -> Any: ...


def _utc_offset(tz: tzinfo | None) -> str:
    """Current UTC offset in ``+HH:MM`` form, for the model to stamp on datetimes.

    Baked in rather than left to the model: Dublin is +01:00 in summer and
    +00:00 in winter, and a model asked to "use the Dublin offset" from memory
    picks one of them all year — or, as on 2026-07-25, a Moscow +03:00 copied out
    of a routine's prompt.

    This is the polite half only. Telling the model the right offset lowers the
    error rate; it cannot stop a model that stamps another one anyway, and that
    failure is silent (the task is created, two hours early, and only a reminder
    at the wrong hour ever shows it). The enforcing half lives in notion-cli
    (``clock.normalize_offset``, >= 0.2.2), which re-stamps any offset that
    disagrees with the configured zone onto the wall clock the caller wrote.
    """
    raw = datetime.now(tz).strftime("%z") or "+0000"
    return f"{raw[:3]}:{raw[3:]}"


def _preserve_time_of_day(page_id: str, new_date: str) -> str:
    """Splice the task's existing time onto ``new_date`` when only the day moved.

    "Move gym to tomorrow" names no time, so the model emits a bare date — which
    would overwrite a 21:15 slot with a date-only value. The task then drops out
    of the slot pinger and silently stops reminding, while the bot reports the
    move as done. That is the worst failure shape available: a request that looks
    honoured and isn't.

    This is enforced in code rather than by a prompt rule on purpose. An
    instruction moves the failure rate down but leaves it silent, and the only
    signal is a reminder that never arrives; reading the current value makes the
    common phrasing incapable of dropping a slot. The cost is one extra
    ``notion-cli tasks get`` per dated update.

    A read failure PROPAGATES rather than falling through to the bare date: if we
    cannot see the current slot, writing the bare day is precisely the silent
    destruction this exists to prevent, and a loud error is recoverable.

    The spliced tail carries whatever offset Notion holds, which for a row written
    before 2026-07-25 may be Moscow's. That is fine and deliberate: the CLI
    re-stamps it (see ``_utc_offset``), so moving an old task also repairs it.
    """
    if not new_date or not _BARE_DATE_RE.match(new_date):
        return new_date
    current = notion.get_task(page_id)
    existing = current.get("date") or "" if isinstance(current, dict) else ""
    if "T" not in existing:
        return new_date
    return new_date + existing[existing.index("T") :]


def build_system(*, today: date | None = None, tz: tzinfo | None = None) -> str:
    """Build the agent system prompt, injecting today's/tomorrow's date.

    The date is baked in (not a live expression like in n8n) because the prompt
    is constructed per process start; long-running bots rebuild it lazily via
    :class:`~notion.bot.NotionBot` so "today" stays correct across days.

    ``tz`` is the scheduling zone, and passing it is not optional in production:
    the VPS runs on UTC, so a bare ``date.today()`` there tells the model it is
    still yesterday for the first hour of every Dublin day (and for the last
    three hours of every Moscow one). ``datetime.now(None)`` is the host's local
    clock, so the default is byte-for-byte the old behaviour.
    """
    today = today or datetime.now(tz).date()
    tomorrow = today + timedelta(days=1)
    offset = _utc_offset(tz)
    return (
        "You are a Notion task manager assistant for Bogdan's Daily Task Tracker.\n\n"
        f"Today is {today.isoformat()}. \"today\" = {today.isoformat()}, "
        f"\"tomorrow\" = {tomorrow.isoformat()}.\n\n"
        "Database: Daily Task Tracker. Properties:\n"
        "- title: the task title (required).\n"
        "- date: either a bare day (YYYY-MM-DD) or a full ISO 8601 datetime with "
        f"offset (YYYY-MM-DDTHH:MM:SS{offset}). Bogdan's timezone is Europe/Dublin "
        f"and its offset RIGHT NOW is {offset} — always use exactly that offset, "
        "never guess another one. The time you write is his wall clock: an offset "
        "that disagrees with Europe/Dublin is corrected to it, never obeyed, so "
        "the hour you name is the hour that lands.\n"
        f"- status: one of {', '.join(STATUS_VALUES)}.\n"
        f"- type: one or more of {', '.join(TYPE_VALUES)} (exactly these, no "
        "others). It is a multi-select: pass a LIST even for a single type — "
        '["English"], not "English".\n'
        f"- effort: one of {', '.join(EFFORT_VALUES)} (exactly these, no others).\n"
        "- project: the EXACT name of a Project Tracker project. A task must join "
        "a project to count toward that project's Progress. Use list_projects to "
        "get exact names; never invent or approximate one.\n"
        "- dod: Definition of Done — one concrete, checkable sentence describing "
        "what 'finished' means. The house rule is no DoD, no entry: when the user "
        "creates a task without one, ask for it (a single short question), unless "
        "they have already declined.\n\n"
        "TIME OF DAY — read this twice, it is the easiest thing to get wrong:\n"
        f"- \"move gym to 5pm\" → a time was named. Emit the FULL datetime: the "
        f"task's existing day at 17:00 with offset {offset}.\n"
        "- \"move gym to tomorrow\" → NO time was named. Only the day changes; the "
        "existing time of day must be KEPT. Pass the bare date "
        "(YYYY-MM-DD) and update_task preserves the time for you.\n"
        "- A task whose date has no time is invisible to the slot pinger and stops "
        "reminding. Never turn a timed task into a date-only one.\n\n"
        "Tools:\n"
        "- create_task(title, date?, types?, effort?, status?, project?, dod?): add a task.\n"
        "- find_tasks(today?, date?, status?, date_from?, date_to?): list tasks. "
        "For a span of days use date_from/date_to (inclusive, YYYY-MM-DD) instead "
        "of many single-date calls. \"this week\" = date_from Monday .. date_to "
        "Sunday of the current week; \"next week\" = the following Monday..Sunday.\n"
        "- update_task(page_id, title?, status?, date?, types?, effort?, project?, "
        "dod?): edit a task.\n"
        "- complete_task(page_id_or_title): mark a task Done.\n"
        "- archive_task(page_id): soft-delete (archive) a task.\n\n"
        "PROJECT TRACKER (a SEPARATE database — projects, not tasks). Read-only:\n"
        "- list_projects(): every project with its status, deadline and progress.\n"
        "- get_project(page_id): one project in full.\n"
        "Use these to answer questions like «в каком состоянии M1zz1 OS» and to "
        "look up an exact project name before setting a task's project. You CANNOT "
        "create, rename, archive or re-status a project — if the user asks for "
        "that, say plainly that project edits are done in Notion or by the nightly "
        "planner, and offer to create a task instead.\n\n"
        "Rules:\n"
        "1. NEVER invent new option values for type, effort or status. Use only the "
        "listed values, with the exact casing shown.\n"
        "2. type and effort are DIFFERENT properties — do not confuse them.\n"
        "3. Before editing or archiving any task, call find_tasks first to locate it, "
        "then operate by its page id (never guess an id).\n"
        "4. Before archiving, confirm with the user explicitly first.\n"
        "5. Reply concisely in the user's language (Russian or English).\n"
        "6. Confirm every successful change (created, updated, completed, archived) "
        "with a brief summary of what changed.\n"
        "7. When the user is vague about the date, assume today unless they say "
        "otherwise.\n"
        "8. If a tool returns an error, explain it plainly and suggest a fix rather "
        "than retrying blindly.\n"
        "9. If a project name does not resolve, the tool says so and lists the real "
        "names. Show the user that list and ask which one they meant. NEVER pick a "
        "close-looking project yourself — a wrong relation is invisible in Notion "
        "but wrong in every progress number.\n\n"
        "HABIT TRACKER (a SEPARATE database — NOT the task tracker):\n"
        "One row per day with a checkbox per habit. It has its OWN tools; never "
        "use task tools for habits or habit tools for tasks.\n"
        f"Habit slugs (use these exact slugs, never invent): {', '.join(HABIT_SLUGS)}.\n"
        f"Russian → slug mapping: {_HABITS_HINT}.\n"
        "Habit tools:\n"
        "- habits_today(): today's habit row.\n"
        "- check_habit(habit, off=False): tick a habit for today (off=True unticks). "
        "'habit' is a slug from the list above.\n"
        "- habit_stats(days=7): completion % per habit over the last N days "
        "(use days=7 for a week, days=30 for a month).\n"
        "Habit rules: map the Russian name to a slug, confirm briefly in Russian "
        "what you ticked (e.g. «Отметил холодную воду на сегодня»), and keep habit "
        "answers short.\n\n"
        "FORMATTING — write Markdown, never HTML. It is converted for you:\n"
        "- **bold** for headings and for anything the user should catch at a "
        "glance (task titles, times, project names).\n"
        "- A line starting with '- ' for each item in a list. One task per line — "
        "never pack several tasks into a paragraph.\n"
        "- *italic* for asides, `code` for page ids and exact option values, "
        "> for quoting the user back.\n"
        "- Structure over prose: a heading line, then the list. Blank line between "
        "sections. No wall of text.\n"
        "- Do NOT write <b>, <i> or any other HTML tag, and do not escape anything "
        "— write '&' and '<' as themselves. The renderer handles it.\n"
        "- Keep it short. Structure is for scanning, not for filling space."
    )


# Backwards-friendly alias: the system prompt for the current day.
AGENT_SYSTEM = build_system()


def register_tools(agent: _AgentLike) -> None:
    """Register the full task-CRUD toolset on a ``core.agent.Agent``.

    Tools are thin sync wrappers over :mod:`core.notion`; the Agent loop calls
    tools synchronously and JSON-serializes their return value back to the model.
    """

    def create_task(
        title: str,
        date: str = "",
        types: str | list[str] | None = None,
        effort: str = "",
        status: str = "",
        project: str = "",
        dod: str = "",
    ) -> Any:
        """Create a task. Empty strings mean "unset".

        types is a LIST of Type values, e.g. ["English"] or ["IT", "Sport"] — Type
        is a multi-select. A single value may also be given as a plain string
        ("English"); it is read as that one value. An unknown value is an error
        naming the ones that exist, so never invent or abbreviate one.

        date accepts EITHER a bare day ('2026-07-25') OR a full ISO 8601 datetime
        with offset ('2026-07-25T21:15:00+01:00'). Pass the full datetime whenever
        the user names a time of day — a bare date creates a task with no slot,
        which the slot pinger cannot remind about.

        project is the EXACT name of a Project Tracker project (call list_projects
        to get the real names); without it the task counts toward no project's
        progress. An unrecognised name is an error listing the available projects
        — show that list to the user rather than picking one.

        dod is the Definition of Done: one concrete, checkable sentence, stored in
        the page body. Returns the created task as JSON.
        """
        return notion.add_task(
            title,
            date=date or None,
            types=types or None,
            effort=effort or None,
            status=status or None,
            project=project or None,
            dod=dod or None,
        )

    def find_tasks(
        today: bool = False,
        date: str = "",
        status: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> Any:
        """List tasks, optionally filtered by today, a single date, a date range, or status.

        Filters (all YYYY-MM-DD; empty strings mean "unset"):
        - today: only today's tasks.
        - date: one specific day.
        - date_from + date_to: an inclusive range (server-side, safe for a whole
          week/month). For "this week" set date_from = Monday and date_to = Sunday;
          "next week" = the following Monday..Sunday. Use the range, not repeated
          single-date calls.
        - status: not started / in progress / done.

        date_from/date_to cannot be combined with today or date. Use this to locate
        a task's page id before updating, completing or archiving it. Returns a list
        of tasks (each with an id and title).
        """
        return notion.list_tasks(
            today=today,
            date=date or None,
            status=status or None,
            date_from=date_from or None,
            date_to=date_to or None,
        )

    def update_task(
        page_id: str,
        title: str = "",
        status: str = "",
        date: str = "",
        types: str | list[str] | None = None,
        effort: str = "",
        project: str = "",
        dod: str = "",
    ) -> Any:
        """Update fields of an existing task by its page id.

        Empty strings mean "leave unchanged". Returns the updated task as JSON.

        types REPLACES the task's Type multi-select and takes exactly the shape
        create_task's does: a list like ["English"], or a plain string for a
        single value. Omit it to leave the types alone.

        date accepts EITHER a bare day ('2026-07-26') OR a full ISO 8601 datetime
        with offset ('2026-07-26T17:00:00+01:00'):
        - The user named a time ("move gym to 5pm") → pass the FULL datetime.
        - The user named only a day ("move gym to tomorrow") → pass the bare date.
          If the task currently has a time slot, that time is carried over onto
          the new day automatically; you do not need to look it up.

        project is the EXACT name of a Project Tracker project (see list_projects);
        an unrecognised name errors with the available names instead of guessing.
        dod replaces the task's Definition of Done in the page body.
        """
        return notion.update_task(
            page_id,
            title=title or None,
            status=status or None,
            date=_preserve_time_of_day(page_id, date) or None,
            types=types or None,
            effort=effort or None,
            project=project or None,
            dod=dod or None,
        )

    def complete_task(page_id_or_title: str) -> Any:
        """Mark a task Done. Accepts a page id or a fuzzy title match."""
        return notion.complete_task(page_id_or_title)

    def archive_task(page_id: str) -> Any:
        """Archive (soft-delete) a task by its page id. Confirm with the user first."""
        notion.delete_task(page_id)
        return {"archived": page_id}

    # ---- project tracker (a THIRD Notion DB, read-only) ----------------
    #
    # Read-only is a deliberate stopping point, not an unfinished one. A project
    # is a long-lived object that many tasks point at, and every write verb has a
    # silent failure mode from a chat/voice channel: a near-miss name creates a
    # SECOND project rather than editing the first, splitting one project's
    # Progress across two rows with nothing visibly wrong in Notion; a misheard
    # word in `projects update` can re-status or re-deadline a project. Tasks are
    # cheap to fix and Bogdan sees them daily; projects are not and he does not.
    # Project creation already has an owner with full context — the nightly
    # planner — so chat write access would add a second, worse-informed writer for
    # no recurring need. Revisit if Bogdan actually hits the wall.

    def list_projects() -> Any:
        """List every project in the PROJECT TRACKER (a separate DB from tasks).

        Returns each project's id, title, status, priority, deadline and progress.
        Use it to answer "what's the state of <project>" and to get the EXACT
        project name required by create_task/update_task's project argument.
        """
        return notion.list_projects()

    def get_project(page_id: str) -> Any:
        """Get one project by page id (find the id with list_projects first).

        Read-only: there is no tool to create, rename, re-status or archive a
        project from chat. Say so plainly if the user asks.
        """
        return notion.get_project(page_id)

    # ---- habit tracker (a separate Notion DB, separate tools) ----------

    def habits_today() -> Any:
        """Return today's HABIT TRACKER row (which habits are ticked today).

        The habit tracker is a SEPARATE database from tasks. Returns a list of
        flattened rows (usually one), each mapping habit slugs to booleans.
        """
        return notion.list_habits_today()

    def check_habit(habit: str, off: bool = False) -> Any:
        """Tick a habit for today; off=True unticks it. Creates today's row if absent.

        'habit' MUST be one of these slugs (map the user's Russian phrasing to a
        slug first): cold, coffee, daily-analysis, meditation, training, trading,
        water, wake-up, course, book, sleep.
        Russian → slug: холодная вода→cold, кофе→coffee, дневной анализ→daily-analysis,
        медитация→meditation, тренировка→training, трейдинг→trading, вода→water,
        подъём→wake-up, курс→course, книга/чтение→book, сон→sleep.
        """
        return notion.check_habit(habit, off=off)

    def habit_stats(days: int = 7) -> Any:
        """Completion % per habit over the last N days (days=7 week, days=30 month).

        Reads the SEPARATE habit tracker database. A missing day counts as
        not done. Returns a list sorted by completion %.
        """
        return notion.habit_stats(days=days)

    task_tools = (create_task, find_tasks, update_task, complete_task, archive_task)
    project_tools = (list_projects, get_project)
    habit_tools = (habits_today, check_habit, habit_stats)
    for fn in (*task_tools, *project_tools, *habit_tools):
        agent.tool(fn)
