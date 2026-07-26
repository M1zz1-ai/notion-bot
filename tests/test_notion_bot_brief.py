"""notion.brief: the bot half of the day loop — announce, then commit at 10:00.

The Mac's day-close routine RPUSHes one brief per day; everything here is what
the bot owes it back. Every test drives an injected clock and a fake redis;
nothing sleeps, nothing shells out to notion-cli.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from notion import brief as brief_mod
from notion.brief import BriefRunner, brief_for, brief_id, commit_draft, fallback_at

DUBLIN = ZoneInfo("Europe/Dublin")
PLAN_DATE = "2026-07-26"
DAY = date(2026, 7, 26)


class FakeTg:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append(text)
        return []


class FakeState:
    """RedisState stand-in: dedup ledger + the brief list."""

    def __init__(self, brief: Any = None) -> None:
        self.marks: set[str] = set()
        self.briefs: list[Any] = [brief] if brief is not None else []
        self.trims: list[tuple[tuple[str, ...], int]] = []

    def seen(self, item_id: str, *, ledger: str = "seen") -> bool:
        return f"{ledger}:{item_id}" in self.marks

    def mark_seen(self, item_id: str, *, ledger: str = "seen", ttl: int = 0) -> None:
        self.marks.add(f"{ledger}:{item_id}")

    def last_json(self, *parts: str, default: Any = None) -> Any:
        return self.briefs[-1] if self.briefs else default

    def trim_list(self, *parts: str, keep: int = 7) -> None:
        self.trims.append((parts, keep))
        self.briefs = self.briefs[-keep:]


class DownState(FakeState):
    """RedisState degraded: seen() always False, mark_seen() a no-op."""

    def seen(self, item_id: str, *, ledger: str = "seen") -> bool:
        return False

    def mark_seen(self, item_id: str, *, ledger: str = "seen", ttl: int = 0) -> None:
        return None


def make_brief(**over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "day_brief",
        "version": 1,
        "close_date": "2026-07-25",
        "plan_date": PLAN_DATE,
        "fallback_commit_at": f"{PLAN_DATE}T10:00:00+01:00",
        "digest_text": "Вчера закрыто 3 задачи.",
        "questions": ["Что ты сегодня делал?", "Что нужно сделать завтра?"],
        "draft": [
            {
                "action": "add",
                "title": "Wire Tasks Tab Complete Button",
                "date": f"{PLAN_DATE}T13:00:00+01:00",
                "type": ["IT"],
                "effort": "High",
                "project": "M1zz1 OS",
                "dod": "Clicking a task flips it to Done in Notion.",
            },
            {
                "action": "update",
                "page_id": "00000000-0000-0000-0000-000000000000",
                "date": f"{PLAN_DATE}T18:00:00+01:00",
                "reason": "carry-over",
            },
        ],
    }
    payload.update(over)
    return payload


def at(hour: int, minute: int = 0) -> datetime:
    """A moment on PLAN_DATE, expressed in UTC (Dublin is +01:00 in July)."""
    return datetime(2026, 7, 26, hour, minute, tzinfo=DUBLIN).astimezone(timezone.utc)


class RecordingCommit:
    def __init__(self, problems: list[str] | None = None) -> None:
        self.calls: list[list[Any]] = []
        self._problems = problems or []

    def __call__(self, draft: list[Any]) -> tuple[int, list[str]]:
        self.calls.append(draft)
        return len(draft) - len(self._problems), list(self._problems)


def build(tg, state, clock, commit=None, **kw) -> BriefRunner:
    return BriefRunner(
        tg,
        chat_id=1,
        state=state,
        tz=DUBLIN,
        now_fn=lambda: clock[0],
        commit=commit or RecordingCommit(),
        **kw,
    )


# ---- reading the brief -------------------------------------------------


def test_brief_for_matches_the_schema_field() -> None:
    """The live schema names the day ``plan_date``; ``date`` never existed."""
    assert brief_for(FakeState(make_brief()), DAY) is not None


def test_brief_for_accepts_the_legacy_date_field() -> None:
    assert brief_for(FakeState({"date": PLAN_DATE}), DAY) is not None


def test_brief_for_rejects_yesterdays_brief() -> None:
    assert brief_for(FakeState(make_brief(plan_date="2026-07-25")), DAY) is None


def test_brief_for_without_state_or_brief() -> None:
    assert brief_for(None, DAY) is None
    assert brief_for(FakeState(), DAY) is None
    assert brief_for(FakeState("not-an-object"), DAY) is None


def test_fallback_at_reads_the_brief() -> None:
    assert fallback_at(make_brief(), DAY, DUBLIN) == datetime(2026, 7, 26, 10, tzinfo=DUBLIN)


def test_fallback_at_defaults_to_ten_local_when_absent() -> None:
    """A brief missing the field must not disable the 11:00 guarantee."""
    assert fallback_at(make_brief(fallback_commit_at=None), DAY, DUBLIN) == datetime(
        2026, 7, 26, 10, tzinfo=DUBLIN
    )


# ---- announcing --------------------------------------------------------


async def test_announces_digest_and_questions() -> None:
    tg, clock = FakeTg(), [at(0, 5)]
    await build(tg, FakeState(make_brief()), clock).tick()
    assert len(tg.texts) == 2
    assert "Вчера закрыто 3 задачи." in tg.texts[0]
    assert "Что ты сегодня делал?" in tg.texts[1]
    assert "Что нужно сделать завтра?" in tg.texts[1]


async def test_announcement_is_not_repeated_across_restarts() -> None:
    state, tg, clock = FakeState(make_brief()), FakeTg(), [at(0, 5)]
    await build(tg, state, clock).tick()
    clock[0] += timedelta(minutes=1)
    await build(tg, state, clock).tick()  # fresh process, same ledger
    assert len(tg.texts) == 2


async def test_announcement_trims_the_brief_list() -> None:
    """Without LTRIM the list grows one blob per day, forever."""
    state, tg, clock = FakeState(make_brief()), FakeTg(), [at(0, 5)]
    await build(tg, state, clock).tick()
    assert state.trims == [(brief_mod.BRIEF_KEY, brief_mod.KEEP_BRIEFS)]


async def test_no_brief_is_silent_here() -> None:
    """The 'planner did not run' shout belongs to the digest, at the digest hour."""
    tg, clock = FakeTg(), [at(9)]
    await build(tg, FakeState(), clock).tick()
    assert tg.texts == []


async def test_empty_brief_text_is_still_announced_loudly() -> None:
    tg, clock = FakeTg(), [at(0, 5)]
    await build(tg, FakeState(make_brief(digest_text="", questions=[])), clock).tick()
    assert len(tg.texts) == 1 and "brief" in tg.texts[0].lower()


async def test_redis_down_announces_at_most_once_per_process() -> None:
    tg, clock = FakeTg(), [at(0, 5)]
    runner = build(tg, DownState(make_brief()), clock)
    for _ in range(5):
        await runner.tick()
        clock[0] += timedelta(minutes=1)
    assert len(tg.texts) == 2


# ---- the fallback commit ----------------------------------------------


async def test_no_commit_before_the_deadline() -> None:
    commit = RecordingCommit()
    clock = [at(9, 59)]
    await build(FakeTg(), FakeState(make_brief()), clock, commit).tick()
    assert commit.calls == []


async def test_commits_the_draft_as_is_at_the_deadline() -> None:
    commit, tg = RecordingCommit(), FakeTg()
    clock = [at(10, 0)]
    await build(tg, FakeState(make_brief()), clock, commit).tick()
    assert len(commit.calls) == 1
    assert commit.calls[0] == make_brief()["draft"]
    assert "2" in tg.texts[-1]


async def test_commit_happens_once_even_across_restarts() -> None:
    state, commit = FakeState(make_brief()), RecordingCommit()
    clock = [at(10, 1)]
    await build(FakeTg(), state, clock, commit).tick()
    clock[0] += timedelta(minutes=1)
    await build(FakeTg(), state, clock, commit).tick()
    assert len(commit.calls) == 1


async def test_a_reply_cancels_the_fallback() -> None:
    state, commit = FakeState(make_brief()), RecordingCommit()
    clock = [at(0, 5)]
    runner = build(FakeTg(), state, clock, commit)
    await runner.tick()
    runner.note_reply()  # Bogdan answered in the morning
    clock[0] = at(10, 5)
    await runner.tick()
    assert commit.calls == []


async def test_a_reply_survives_a_restart() -> None:
    """The answer lives in the ledger, not in the process that heard it."""
    state, commit = FakeState(make_brief()), RecordingCommit()
    clock = [at(0, 5)]
    build(FakeTg(), state, clock, commit).note_reply()
    clock[0] = at(10, 5)
    await build(FakeTg(), state, clock, commit).tick()
    assert commit.calls == []


async def test_yesterdays_reply_does_not_cancel_todays_fallback() -> None:
    state, commit = FakeState(make_brief()), RecordingCommit()
    clock = [datetime(2026, 7, 25, 20, tzinfo=DUBLIN).astimezone(timezone.utc)]
    build(FakeTg(), state, clock, commit).note_reply()
    clock[0] = at(10, 5)
    await build(FakeTg(), state, clock, commit).tick()
    assert len(commit.calls) == 1


async def test_commit_problems_are_reported_out_loud() -> None:
    tg = FakeTg()
    commit = RecordingCommit(problems=["#1 carry-over: notion-cli failed (1)"])
    await build(tg, FakeState(make_brief()), [at(10, 0)], commit).tick()
    assert "notion-cli failed" in tg.texts[-1]


async def test_empty_draft_is_reported_as_no_plan() -> None:
    tg = FakeTg()
    await build(tg, FakeState(make_brief(draft=[])), [at(10, 0)], RecordingCommit()).tick()
    assert "no plan for today" in tg.texts[-1].lower()


# ---- a second brief for the same day -----------------------------------


def test_brief_id_is_content_derived_and_stable() -> None:
    """Identity comes from the payload, not from the position in the list."""
    assert brief_id(make_brief()) == brief_id(make_brief())
    assert brief_id(make_brief()) != brief_id(make_brief(draft=[]))


async def test_a_superseding_brief_is_announced_before_it_is_committed() -> None:
    """The 2026-07-26 bug: a brief pushed AFTER the announce was committed unseen.

    19:21 pushed a 3-item draft, 00:00 announced it, 00:03 pushed a 6-item one —
    and the 10:00 fallback read the newest element, committing a plan Bogdan had
    never been shown.
    """
    first = make_brief(digest_text="Draft of three.", draft=[{"action": "add", "title": "A"}])
    second = make_brief(digest_text="Draft of six.", draft=[{"action": "add", "title": "B"}])
    state, tg, commit, clock = FakeState(first), FakeTg(), RecordingCommit(), [at(0, 0)]
    runner = build(tg, state, clock, commit)
    await runner.tick()
    state.briefs.append(second)  # 00:03: the scheduled day-close, after the announce
    clock[0] = at(0, 3)
    await runner.tick()
    clock[0] = at(10, 0)
    await runner.tick()
    assert commit.calls == [second["draft"]]
    assert "Draft of six." in tg.texts[3], "the committed draft must have been announced"
    assert "committing" in tg.texts[-1].lower()


async def test_a_superseding_brief_is_flagged_as_an_update() -> None:
    """A silent swap is the divergence again, one step later — say it out loud."""
    state, tg, clock = FakeState(make_brief()), FakeTg(), [at(0, 0)]
    runner = build(tg, state, clock)
    await runner.tick()
    state.briefs.append(make_brief(digest_text="Rewritten."))
    clock[0] = at(0, 3)
    await runner.tick()
    assert "replaces" in tg.texts[2].lower()
    assert "Rewritten." in tg.texts[3]


async def test_an_unchanged_brief_is_never_re_announced() -> None:
    """A re-run that pushes the same payload is not new information."""
    state, tg, clock = FakeState(make_brief()), FakeTg(), [at(0, 0)]
    runner = build(tg, state, clock)
    await runner.tick()
    state.briefs.append(make_brief())
    clock[0] = at(0, 3)
    await runner.tick()
    assert len(tg.texts) == 2


async def test_a_normal_day_announces_once_and_commits_once() -> None:
    state, tg, commit, clock = FakeState(make_brief()), FakeTg(), RecordingCommit(), [at(0, 0)]
    runner = build(tg, state, clock, commit)
    for moment in (at(0, 0), at(0, 1), at(9, 59), at(10, 0), at(10, 1)):
        clock[0] = moment
        await runner.tick()
    assert len(commit.calls) == 1
    assert len([t for t in tg.texts if "Вчера закрыто 3 задачи." in t]) == 1


async def test_a_reply_stands_the_fallback_down_for_a_new_brief_too() -> None:
    """Answered means answered: a later brief is shown, never written."""
    state, tg, commit, clock = FakeState(make_brief()), FakeTg(), RecordingCommit(), [at(0, 0)]
    runner = build(tg, state, clock, commit)
    await runner.tick()
    runner.note_reply()
    state.briefs.append(make_brief(digest_text="Rewritten after the reply."))
    clock[0] = at(10, 5)
    await runner.tick()
    assert commit.calls == []
    assert "settled" in tg.texts[2].lower()
    assert "Rewritten after the reply." in tg.texts[3]


async def test_a_brief_arriving_after_the_deadline_is_shown_but_not_committed() -> None:
    """Late is late: announcing it as actionable would be the divergence inverted."""
    late = make_brief(digest_text="Too late.", draft=[{"action": "add", "title": "L"}])
    state, tg, commit, clock = FakeState(make_brief()), FakeTg(), RecordingCommit(), [at(0, 0)]
    runner = build(tg, state, clock, commit)
    await runner.tick()
    clock[0] = at(10, 0)
    await runner.tick()  # commits the announced brief
    state.briefs.append(late)
    clock[0] = at(10, 30)
    await runner.tick()
    assert commit.calls == [make_brief()["draft"]]
    assert "settled" in tg.texts[3].lower()
    assert "Too late." in tg.texts[4]


# ---- draft -> notion-cli ----------------------------------------------


class FakeNotion:
    def __init__(self, fail_on: str | None = None) -> None:
        self.added: list[tuple[str, dict[str, Any]]] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self._fail_on = fail_on

    def add_task(self, title: str, **kw: Any) -> dict[str, Any]:
        if self._fail_on == "add":
            raise RuntimeError("notion-cli add failed (1)")
        self.added.append((title, kw))
        return {"id": "new"}

    def update_task(self, page_id: str, **kw: Any) -> dict[str, Any]:
        if self._fail_on == "update":
            raise RuntimeError("notion-cli update failed (1)")
        self.updated.append((page_id, kw))
        return {"id": page_id}


def test_commit_draft_maps_add_and_update() -> None:
    fake = FakeNotion()
    committed, problems = commit_draft(make_brief()["draft"], notion_mod=fake)
    assert (committed, problems) == (2, [])
    title, kw = fake.added[0]
    assert title == "Wire Tasks Tab Complete Button"
    assert kw["types"] == ["IT"] and kw["effort"] == "High"
    assert kw["project"] == "M1zz1 OS" and kw["dod"].startswith("Clicking")
    page_id, upd = fake.updated[0]
    assert page_id.startswith("00000000") and upd == {"date": f"{PLAN_DATE}T18:00:00+01:00"}


def test_commit_draft_accepts_a_scalar_type() -> None:
    fake = FakeNotion()
    commit_draft([{"action": "add", "title": "T", "type": "Sport"}], notion_mod=fake)
    assert fake.added[0][1]["types"] == ["Sport"]


def test_commit_draft_drops_an_update_without_page_id() -> None:
    """Rule 2 of the routine: a carry-over without a page id is a clone waiting
    to happen — drop it rather than turn it into a new row."""
    fake = FakeNotion()
    committed, problems = commit_draft([{"action": "update", "date": "x"}], notion_mod=fake)
    assert (committed, fake.added, fake.updated) == (0, [], [])
    assert problems and "page_id" in problems[0]


def test_commit_draft_drops_unknown_actions() -> None:
    fake = FakeNotion()
    committed, problems = commit_draft(
        [{"action": "delete", "page_id": "x"}, "junk"], notion_mod=fake
    )
    assert committed == 0 and len(problems) == 2


def test_commit_draft_survives_one_failing_entry() -> None:
    fake = FakeNotion(fail_on="add")
    committed, problems = commit_draft(make_brief()["draft"], notion_mod=fake)
    assert committed == 1 and len(problems) == 1
    assert fake.updated, "the carry-over must still be moved"


def test_commit_draft_handles_an_absent_draft() -> None:
    assert commit_draft(None, notion_mod=FakeNotion()) == (0, [])
