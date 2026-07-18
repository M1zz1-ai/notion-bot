"""Notion-bot capabilities: core.notion CRUD exposed as agent tools.

Each capability is a small callable registered on a
``core.openai_agent.OpenAIAgent``. The tools shell out via :mod:`core.notion`
(which wraps the external ``notion-cli`` — we do NOT reimplement Notion auth
here).

The agent option enums are aligned to what ``notion-cli`` actually accepts
(Type: Trading/Gym/IT/Sport; Effort: Low/Medium/High; Status: not started/in
progress/done). Adjust them to match your own Notion database schema.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Protocol

from core import notion

# LLM brain: a fast/cheap OpenAI reasoning model handles the tool-calling
# task/habit CRUD loop; bump to a stronger model via ``NOTION_MODEL`` if
# accuracy (e.g. week-range date math) regresses.
NOTION_MODEL = os.getenv("NOTION_MODEL", "gpt-5.4-mini")

# notion-cli's accepted option values — the model must never invent others.
TYPE_VALUES = ("Trading", "Gym", "IT", "Sport")
EFFORT_VALUES = ("Low", "Medium", "High")
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


def build_system(*, today: date | None = None) -> str:
    """Build the agent system prompt, injecting today's/tomorrow's date.

    The date is baked in at construction time; long-running bots rebuild the
    prompt lazily via :class:`~notion.bot.NotionBot` so "today" stays correct
    across days.
    """
    today = today or date.today()
    tomorrow = today + timedelta(days=1)
    return (
        "You are a Notion task manager assistant for your Daily Task Tracker.\n\n"
        f"Today is {today.isoformat()}. \"today\" = {today.isoformat()}, "
        f"\"tomorrow\" = {tomorrow.isoformat()}.\n\n"
        "Database: Daily Task Tracker. Properties:\n"
        "- title: the task title (required).\n"
        "- date: YYYY-MM-DD.\n"
        f"- status: one of {', '.join(STATUS_VALUES)}.\n"
        f"- type: one or more of {', '.join(TYPE_VALUES)} (exactly these, no others).\n"
        f"- effort: one of {', '.join(EFFORT_VALUES)} (exactly these, no others).\n\n"
        "Tools:\n"
        "- create_task(title, date?, types?, effort?, status?): add a task.\n"
        "- find_tasks(today?, date?, status?, date_from?, date_to?): list tasks. "
        "For a span of days use date_from/date_to (inclusive, YYYY-MM-DD) instead "
        "of many single-date calls. \"this week\" = date_from Monday .. date_to "
        "Sunday of the current week; \"next week\" = the following Monday..Sunday.\n"
        "- update_task(page_id, title?, status?, date?, type_?, effort?): edit a task.\n"
        "- complete_task(page_id_or_title): mark a task Done.\n"
        "- archive_task(page_id): soft-delete (archive) a task.\n\n"
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
        "than retrying blindly.\n\n"
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
        "answers short."
    )


# Backwards-friendly alias: the system prompt for the current day.
AGENT_SYSTEM = build_system()


def register_tools(agent: _AgentLike) -> None:
    """Register the full task-CRUD toolset on a ``core.openai_agent.OpenAIAgent``.

    Tools are thin sync wrappers over :mod:`core.notion`; the Agent loop calls
    tools synchronously and JSON-serializes their return value back to the model.
    """

    def create_task(
        title: str,
        date: str = "",
        types: list[str] | None = None,
        effort: str = "",
        status: str = "",
    ) -> Any:
        """Create a task. types is a list of Type values; date is YYYY-MM-DD.

        Empty strings mean "unset". Returns the created task as JSON.
        """
        return notion.add_task(
            title,
            date=date or None,
            types=types or None,
            effort=effort or None,
            status=status or None,
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
        type_: str = "",
        effort: str = "",
    ) -> Any:
        """Update fields of an existing task by its page id.

        Empty strings mean "leave unchanged". Returns the updated task as JSON.
        """
        return notion.update_task(
            page_id,
            title=title or None,
            status=status or None,
            date=date or None,
            type_=type_ or None,
            effort=effort or None,
        )

    def complete_task(page_id_or_title: str) -> Any:
        """Mark a task Done. Accepts a page id or a fuzzy title match."""
        return notion.complete_task(page_id_or_title)

    def archive_task(page_id: str) -> Any:
        """Archive (soft-delete) a task by its page id. Confirm with the user first."""
        notion.delete_task(page_id)
        return {"archived": page_id}

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
    habit_tools = (habits_today, check_habit, habit_stats)
    for fn in (*task_tools, *habit_tools):
        agent.tool(fn)
