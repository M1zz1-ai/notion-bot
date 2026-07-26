# notion-bot

A Telegram bot that replans one person's day. It **moves** unfinished tasks
instead of cloning them, **shows** you the plan before it commits it, and
**refuses** to call a broken planner a rest day.

📊 **Case study: [a day that replans itself](https://m1zz1-ai.github.io/notion-task-bot/)**
— the failure, the fix, and the architecture in eight screens. One self-contained
HTML file, served from [`docs/`](docs/); open it locally if you would rather not
leave the repo.

---

## The failure it was built from

The previous generator re-created every unfinished task each morning and left the
original open. Two weeks in, a tracker holding 17 tasks a day held 187 — 1,234
clone rows, up to 66 copies of a single task. A list that long cannot be worked,
so it stops being opened, and the plan quietly stops existing.

Every design decision below is downstream of that.

| The rule | What it prevents |
|---|---|
| One row, one identity — a carry-over is an `update`, never an `add` | The 187 |
| A commit is gated on the hash of the brief that was announced | Committing a plan you were never shown |
| An empty day and a dead planner are different messages | Silence being read as rest |
| Every write leaves through one CLI | Two writers fighting over the same tracker |

## What it does

One systemd unit, four loops running side by side:

- **long-poll** — chat and voice. Whisper transcribes, an OpenAI agent picks from
  11 typed tools, and the CLI executes. The model never sees a Notion token.
- **digest** — at a wall-clock hour (default 11:00), the day laid out slot by slot.
- **pinger** — a 60-second tick that reminds you at the slot, keyed by
  `page_id + slot` so a restart re-sends nothing and a reschedule re-arms.
- **brief runner** — consumes one plan draft per day, announces it, and commits it
  if you have not answered by the fallback hour.

Everything that touches Notion — the model, a timer, a committed draft — leaves
through `notion-cli`. Enum validation, timezone stamping and DoD placement happen
in exactly one place because there is exactly one door.

---

## Three ways to run it

The honest answer to "can I use this?" depends on how much of the surrounding
machinery you have. All three are real; only the third is the whole product.

### 1. Dry — clone and read it (no credentials, no accounts)

Everything below runs on a fresh clone with nothing configured:

```bash
uv sync --dev && uv run pytest -q && uv run ruff check .
```

306 tests, **zero network calls**. Telegram, Notion and OpenAI are all faked at
the boundary, so CI is green without a single secret — and so is your laptop.
That is deliberate: the count is not the claim, the determinism is.

(The case study says 197. That is this repo's notion-specific subset —
`tests/test_notion*.py` — counted in the monorepo it is exported from. The other
109 cover the shared `core/` modules that came along with it.)

What you can exercise dry: the slot arithmetic and DST handling
(`notion/timeslots.py`), the brief identity hash and the announce/commit gating
(`notion/brief.py`), the "planner did not run" guard (`notion/digest.py`), the
ping dedup keys (`notion/pinger.py`), the tool schemas and enum validation
(`notion/tools.py`, `core/notion.py`).

What does **not** work dry: anything that reaches a real service. There is no
demo mode and no fixture Notion — this is a bot, not a library, and pretending
otherwise would just waste your afternoon.

### 2. Standalone — your Telegram, your Notion

Bring the four things in [What you have to bring yourself](#what-you-have-to-bring-yourself),
fill `.env`, run `python -m notion --check`, then start the unit.

You get: conversational task and habit CRUD by text or voice, the daily digest,
and slot reminders that survive a restart.

You do **not** get the headline feature. Nightly replanning needs a *planner* —
something that decides, each midnight, what carries over and what the next day
should look like. This repo is the half that talks, listens and writes; it is
not the half that thinks about your day. Without a producer pushing a brief, the
brief runner idles and the digest tells you, loudly, that the planner did not
run. That message is correct: it did not.

If you want the full loop standalone, write your own producer against
[the brief contract](#the-brief-contract) below. It is one JSON object per day
on a redis list — deliberately a data contract and not a plugin API, so the
planner can be a cron script, another bot, or a person with `redis-cli`.

### 3. Connected — inside an agent office with memory

This is how it actually runs. The planner is a scheduled Claude Code routine
(`day-close`, 00:00) on a second machine, and the split is the point: **the Mac
thinks, the VPS speaks.**

The routine writes nothing to Notion. It reads the day that ended and emits one
brief; the bot owns every write. Two writers is how a tracker starts fighting
itself.

What the office and the memory add on top of a bare producer:

- **Yesterday, as evidence rather than memory.** The planner reads the closed
  day out of Notion plus the office's own run logs, so "what actually happened"
  is measured, not recalled.
- **Continuity across days.** A long-lived vault (`brain`) carries active plans,
  project state and standing decisions, so today's draft knows what last week
  committed to and does not re-propose finished work.
- **Judgment about what belongs in a day.** Work that is no longer yours gets
  filtered before it reaches the draft — a rule that lives in the planner, where
  taste belongs, and never in the bot.
- **A tone that is yours.** `digest_text` arrives ready to send, written in your
  language and register, because the routine that wrote it has read a lot of you.
- **One voice.** The routine never sends Telegram messages of its own. Two
  machines, one identity — everything reaches you as the bot.

None of that lives in this repository, and none of it is required to read it.
The seam is the brief: everything above produces one JSON object, and everything
in here consumes it.

---

## What you have to bring yourself

1. **`notion-cli`** — the single write path. It owns your Notion integration
   token and your task/habit database ids; this bot shells out to it and parses
   `--json`. **It is not published, and this repo does not reimplement it.** You
   need a CLI on the host at `~/bin/notion-cli` (or pass `cli=`) exposing
   `tasks add|list|update|complete|archive` and `habits …` with `--json` output
   and `--version`. `core/notion.py` documents the exact argv and enforces a
   version floor, so a stale CLI fails the preflight instead of half-working.
2. **A Telegram bot token** from @BotFather, and your numeric chat id.
3. **An OpenAI API key** — one key covers both the agent and Whisper.
4. **A redis** for the ledgers. Without one the bot still runs, but reminders
   degrade from exactly-once to at-most-once.

**Honest scope.** This is one person's tool, wired to one Notion workspace's
schema. A stranger cannot clone it and be running in ten minutes, and this
README is not going to imply otherwise. Read it as a reference implementation of
the pattern — idempotent replanning, content-hash gating, human-in-the-loop
commits, one write path — rather than as an app to install.

## Configuration

Copy `.env.example` to `.env` and fill it in. If no local `.env` exists, config
falls back to `~/.config/m1zz1/.env`, which is how this runs alongside its
siblings on the author's host; either path works and the local one wins.

```bash
uv run python -m notion --check       # validate config + notion-cli version, exit
uv run python -m notion --digest-once # send one digest and exit
uv run python -m notion               # the unit: all four loops
```

`TIMEZONE` is a ZoneInfo name and never a fixed offset — the digest and the
pings fire on a wall clock, so the zone has to carry its own DST.

## The brief contract

One JSON object per day, `RPUSH`ed to `m1zz1:notion:brief`. The bot reads with
`LRANGE` and never pops: the morning digest reads the same element as proof the
planner ran.

```json
{
  "type": "day_brief",
  "version": 1,
  "close_date": "2026-01-14",
  "plan_date": "2026-01-15",
  "generated_at": "2026-01-15T00:00:11+00:00",
  "fallback_commit_at": "2026-01-15T10:00:00+00:00",
  "digest_text": "sent as-is, before the questions",
  "questions": ["What did you do today?", "What needs doing tomorrow?"],
  "draft": [
    {"action": "add", "title": "Draft release notes", "date": "2026-01-15T13:00:00",
     "type": ["IT"], "effort": "High", "project": "Example", "dod": "Notes published."},
    {"action": "update", "page_id": "<notion-page-id>",
     "date": "2026-01-15T18:00:00", "reason": "carry-over"}
  ]
}
```

Three things this shape is load-bearing about:

- `generated_at` and `fallback_commit_at` carry an offset because they are
  instants read on a host whose clock is not yours. Everything under `draft` is
  local wall-clock with no offset — the CLI stamps the zone, once.
- A carry-over is `action: "update"` with a `page_id`. An `add` without one is
  the 187 bug, re-entering through the front door.
- Push a second brief for the same day and it is announced as an explicit
  update. Announce and commit name the *same* brief by content hash, so a
  superseding draft can never be committed under cover of an older announcement.

## Tests

```bash
uv run pytest -q          # the whole suite, no network, no secrets
uv run ruff check .
```

`notion/live_smoke.py` is the only thing that talks to real services. It is
gated on credentials being present and skips loudly when they are not.

## Layout

```
notion/      the bot: bot, brief, digest, pinger, timeslots, tools
core/        shared with its siblings: config, errors, notion, openai_agent,
             scheduler, state, stt, tg, tgfmt
tests/       the suite above
deploy/      systemd unit template
docs/        the case study
```

`core/` is a subset of a monorepo's shared package, exported by script rather
than by hand — the modules other bots need (spreadsheets, image generation) are
deliberately not here, so cloning this does not drag in pandas.

## License

MIT. See [LICENSE](LICENSE).
