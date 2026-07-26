"""The bot half of the day loop: consume the Mac's brief, then guarantee a plan.

The Mac's ``day-close`` routine (00:00) RPUSHes ONE brief per day to
``m1zz1:notion:brief`` and deliberately writes nothing to Notion — two writers
is how a task tracker starts fighting itself. Everything the brief promises
Bogdan therefore has to happen here:

1. **Open the conversation.** Send ``digest_text``, then the two ``questions``.
2. **Guarantee a plan.** If Bogdan has not answered by ``fallback_commit_at``
   (10:00 in ``TIMEZONE``), commit ``draft`` AS IS, so 11:00 always has a plan.
   The plan is an offer; silence is a conscious decision, and this is what
   makes silence a valid answer instead of an empty day.
3. **Keep the list finite.** The brief is read with LRANGE, never popped — the
   morning digest reads the same element as proof the planner ran. Nothing would
   ever delete it, so the list is trimmed to :data:`KEEP_BRIEFS` once a day.

Four dedup keys carry the whole design, all in the ``brief`` ledger and all
dated, so a systemd restart is not a new day:

* ``announce:<day>:<brief_id>`` — THIS brief was shown to Bogdan. The commit is
  gated on it, which is what makes "never commit a draft he was not shown" a
  property of the code rather than a coincidence of timing.
* ``announce:<day>`` — some brief was shown today, so the next one is an update
  and is announced as one.
* ``answer:<day>``  — Bogdan spoke to the bot; the fallback stands down.
* ``commit:<day>``  — marked BEFORE writing to Notion. A crash mid-commit
  leaves a partial plan and a loud message; retrying would duplicate rows, and
  duplicated rows are the disease this whole loop was built to cure.

A day can hold more than one brief — a manual ``day-close``, a re-run, a retry —
and on the night of 2026-07-26 it did: a 3-item draft was announced at 00:00 and
a 6-item one landed at 00:03. Because the commit read "the newest element of the
list", the 10:00 fallback was one unanswered morning away from writing a plan
Bogdan had never seen. The fix is not "announce once per day"; it is that
announce and commit name the SAME brief by :func:`brief_id`, and a superseding
brief is announced as an explicit update rather than swapped in silently.

A missing brief is silent HERE on purpose: the digest already shouts about it at
the digest hour (``digest.PLANNER_MISSING``), and one alarm is a signal while
two are noise.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any, Protocol

from core import notion
from core.scheduler import run_interval

from .timeslots import parse_slot

logger = logging.getLogger(__name__)

# RedisState prefixes its namespace, so this resolves to ``m1zz1:notion:brief``
# — the key ``~/bin/report_transport`` RPUSHes to from the Mac.
BRIEF_KEY = ("notion", "brief")
KEEP_BRIEFS = 7
LEDGER = "brief"
# Must outlive one loop (announce 00:00 -> commit 10:00) with room for a late run.
TTL = 72 * 3600
TICK_SECONDS = 60
DEFAULT_FALLBACK_HOUR = 10

# Update fields a carry-over may legitimately carry. ``type`` is absent: the
# draft schema makes it a list and ``notion-cli tasks update`` takes one value.
UPDATE_FIELDS = ("title", "status", "date", "effort", "project", "dod")

BRIEF_EMPTY = (
    "⚠️ <b>Today's brief arrived empty.</b>\n"
    "The planner ran, but sent neither a digest nor questions — "
    "check the day-close routine."
)
COMMITTED = (
    "🗓 No answer by {deadline} — committing last night's draft as is: "
    "{n} task(s) for today.\nYou can change the plan right here, in a normal message."
)
COMMITTED_NOTHING = (
    "🗓 No answer by {deadline}, and the draft turned out to hold no tasks at all — "
    "there is no plan for today. That is a planner failure, not a day off."
)
COMMIT_PROBLEMS = "⚠️ Not everything was written:\n{problems}"
SUPERSEDED = (
    "🔁 <b>The planner sent an updated brief for today.</b>\n"
    "It replaces the earlier one — this is the version that gets committed "
    "if you stay silent."
)
SUPERSEDED_SETTLED = (
    "🔁 <b>The planner sent an updated brief for today — too late to be used.</b>\n"
    "Today's plan is already settled; nothing below is written to Notion "
    "automatically."
)


class _TgLike(Protocol):
    async def send_text(self, text: str, chat_id: int | None = ..., **kw: Any) -> Any: ...


class _StateLike(Protocol):
    def seen(self, item_id: str, *, ledger: str = ...) -> bool: ...
    def mark_seen(self, item_id: str, *, ledger: str = ..., ttl: int = ...) -> None: ...
    def last_json(self, *parts: str, default: Any = ...) -> Any: ...
    def trim_list(self, *parts: str, keep: int = ...) -> None: ...


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---- reading the brief -------------------------------------------------


def brief_date(brief: dict[str, Any]) -> str:
    """The day a brief plans for.

    ``plan_date`` is the live schema (version 1); ``date`` is accepted because
    the first readers of this key were written against a field that the routine
    never actually emitted — matching only on it made every morning report a
    dead planner.
    """
    for field in ("plan_date", "date"):
        value = brief.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def brief_id(brief: dict[str, Any]) -> str:
    """A stable identity for ONE brief payload.

    Derived from the content, never from the position in the redis list — the
    list position is exactly what let "the brief we announced" and "the brief we
    commit" drift apart. Two consequences are the point:

    * a brief re-pushed unchanged (a retry, a re-run of ``day-close``) has the
      SAME id, so it is not new information and is not re-announced;
    * a brief with a different draft has a different id, so it can never be
      committed under the identity of the one Bogdan was shown.

    If the Mac's routine ever stamps its own id, honour it here instead of
    hashing — but only if that id changes whenever the draft does.
    """
    blob = json.dumps(brief, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def brief_for(state: _StateLike | None, day: dt.date) -> dict[str, Any] | None:
    """The newest brief if it plans ``day``, else None (never raises).

    Read-only: consuming the brief would erase the digest's only evidence that
    the planner ran.
    """
    if state is None:
        return None
    brief = state.last_json(*BRIEF_KEY)
    if not isinstance(brief, dict):
        return None
    return brief if brief_date(brief).startswith(day.isoformat()) else None


def fallback_at(brief: dict[str, Any], day: dt.date, tz: dt.tzinfo) -> dt.datetime:
    """When the draft gets committed unanswered.

    Falls back to 10:00 local: a brief that forgot the field must not silently
    disable the guarantee that 11:00 has a plan.
    """
    moment = parse_slot(brief.get("fallback_commit_at"), tz)
    if moment is not None:
        return moment
    return dt.datetime.combine(day, dt.time(DEFAULT_FALLBACK_HOUR), tzinfo=tz)


# ---- writing the draft -------------------------------------------------


def _as_list(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value] if value else None
    if isinstance(value, list):
        return [str(v) for v in value] or None
    return None


def _commit_entry(entry: dict[str, Any], notion_mod: Any) -> None:
    """Write ONE draft entry. Raises ValueError on a malformed entry."""
    action = str(entry.get("action") or "").lower()
    if action == "add":
        title = entry.get("title")
        if not title:
            raise ValueError('"add" entry without a title')
        notion_mod.add_task(
            str(title),
            date=entry.get("date"),
            types=_as_list(entry.get("type")),
            effort=entry.get("effort"),
            project=entry.get("project"),
            dod=entry.get("dod"),
        )
        return
    if action == "update":
        page_id = entry.get("page_id")
        if not page_id:
            # Rule 2 of the routine: a carry-over with no page_id is a clone
            # waiting to happen. Drop it; never "helpfully" add it instead.
            raise ValueError('"update" entry without a page_id')
        fields = {name: entry.get(name) for name in UPDATE_FIELDS}
        notion_mod.update_task(str(page_id), **{k: v for k, v in fields.items() if v})
        return
    raise ValueError(f"unknown action {action!r}")


def commit_draft(draft: list[Any] | None, *, notion_mod: Any = notion) -> tuple[int, list[str]]:
    """Write every draft entry to Notion; return ``(committed, problems)``.

    One bad or failing entry never sinks the rest — a plan with two of three
    tasks is a plan; an exception here would be an empty day.
    """
    committed = 0
    problems: list[str] = []
    for index, entry in enumerate(draft or []):
        if not isinstance(entry, dict):
            problems.append(f"#{index}: not an object")
            continue
        label = entry.get("title") or entry.get("page_id") or "?"
        try:
            _commit_entry(entry, notion_mod)
        except Exception as exc:  # noqa: BLE001 - per-entry isolation is the point
            logger.warning("draft entry #%s (%s) failed: %s", index, label, exc)
            problems.append(f"#{index} {label}: {exc}")
            continue
        committed += 1
    logger.info("draft committed: %s ok, %s problem(s)", committed, len(problems))
    return committed, problems


# ---- the runner --------------------------------------------------------


class BriefRunner:
    """Ticks once a minute: announce today's brief, then hold the 10:00 line.

    Args:
        telegram: a ``core.tg.TelegramClient`` (or compatible).
        chat_id: Bogdan's chat — the only voice of this loop.
        state: a ``core.state.RedisState`` (or compatible ledger + list reader).
        tz: zone the brief's wall-clock times are read and rendered in.
        now_fn: injectable clock (tests).
        commit: injectable writer, ``draft -> (committed, problems)``.
        tick_seconds: loop period.
    """

    def __init__(
        self,
        telegram: _TgLike,
        *,
        chat_id: int,
        state: _StateLike,
        tz: dt.tzinfo,
        now_fn: Callable[[], dt.datetime] = _utc_now,
        commit: Callable[[list[Any]], tuple[int, list[str]]] = commit_draft,
        tick_seconds: int = TICK_SECONDS,
    ) -> None:
        self._tg = telegram
        self._chat_id = chat_id
        self._state = state
        self._tz = tz
        self._now = now_fn
        self._commit = commit
        self._tick_seconds = tick_seconds
        self._local: set[str] = set()

    # ---- dedup ---------------------------------------------------------

    def _seen(self, key: str) -> bool:
        """True per this process OR the ledger.

        ``RedisState.seen`` answers False when redis is down ("treat as
        unseen") — right for mail, catastrophic here, where it would re-announce
        every tick and re-commit the draft every minute. The in-process set
        degrades an outage to at-most-once-per-process.
        """
        return key in self._local or self._state.seen(key, ledger=LEDGER)

    def _mark(self, key: str) -> None:
        self._local.add(key)
        self._state.mark_seen(key, ledger=LEDGER, ttl=TTL)

    def _today(self) -> dt.date:
        return self._now().astimezone(self._tz).date()

    def note_reply(self) -> None:
        """Record that Bogdan answered today — today's fallback stands down.

        Any message counts: the contract is "has not answered by 10:00", and
        the bot cannot know which sentence was the plan. Wired from
        :class:`~notion.bot.NotionBot` for the owner chat only.
        """
        self._mark(f"answer:{self._today().isoformat()}")

    # ---- tick ----------------------------------------------------------

    async def tick(self) -> None:
        """Show today's brief once, then commit THAT brief once at the deadline."""
        now = self._now()
        day = self._today()
        brief = brief_for(self._state, day)
        if brief is None:
            return  # the digest owns the "planner did not run" shout

        iso = day.isoformat()
        answer_key, commit_key = f"answer:{iso}", f"commit:{iso}"
        shown_key = f"announce:{iso}:{brief_id(brief)}"
        if not self._seen(shown_key):
            settled = self._seen(answer_key) or self._seen(commit_key)
            await self._announce(brief, superseded=self._seen(f"announce:{iso}"), settled=settled)
            self._mark(shown_key)
            self._mark(f"announce:{iso}")
            self._state.trim_list(*BRIEF_KEY, keep=KEEP_BRIEFS)

        # Re-read: Bogdan may have answered while the announcement was in flight.
        if self._seen(answer_key) or self._seen(commit_key):
            return
        if not self._seen(shown_key):
            return  # never commit a draft he was not shown
        deadline = fallback_at(brief, day, self._tz)
        if now < deadline:
            return
        self._mark(commit_key)  # before the write: a retry would duplicate rows
        await self._commit_draft(brief, deadline)

    async def run(self, *, alerter: Any | None = None, max_cycles: int | None = None) -> None:
        """Tick forever, each tick wrapped in core resilience."""
        await run_interval(
            self.tick,
            self._tick_seconds,
            alerter=alerter or self._tg,
            label="day brief",
            max_cycles=max_cycles,
        )

    # ---- messages ------------------------------------------------------

    async def _announce(
        self, brief: dict[str, Any], *, superseded: bool = False, settled: bool = False
    ) -> None:
        """Open the day: the digest text, then the questions as one message.

        A brief that is not the first one of the day is prefixed with what it
        actually is. ``settled`` (the plan is already committed, or Bogdan has
        answered) says the draft below will NOT be written — announcing it as if
        it still counted would be the same shown-vs-written divergence, inverted.
        """
        if superseded:
            await self._tg.send_text(
                SUPERSEDED_SETTLED if settled else SUPERSEDED, chat_id=self._chat_id
            )
        text = str(brief.get("digest_text") or "").strip()
        questions = [
            q.strip() for q in (brief.get("questions") or []) if isinstance(q, str) and q.strip()
        ]
        if not text and not questions:
            logger.warning("brief for %s has no digest_text and no questions", brief_date(brief))
            await self._tg.send_text(BRIEF_EMPTY, chat_id=self._chat_id)
            return
        if text:
            await self._tg.send_text(text, chat_id=self._chat_id)
        if questions:
            await self._tg.send_text("\n".join(questions), chat_id=self._chat_id)

    async def _commit_draft(self, brief: dict[str, Any], deadline: dt.datetime) -> None:
        """Write the draft as is and say so — silently committing is not a plan."""
        draft = brief.get("draft") or []
        committed, problems = await asyncio.to_thread(self._commit, draft)
        when = deadline.astimezone(self._tz).strftime("%H:%M")
        template = COMMITTED if committed else COMMITTED_NOTHING
        lines = [template.format(deadline=when, n=committed)]
        if problems:
            lines.append(COMMIT_PROBLEMS.format(problems="\n".join(f"• {p}" for p in problems)))
        await self._tg.send_text("\n\n".join(lines), chat_id=self._chat_id)
