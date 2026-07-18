"""Interval loop: runs N cycles, survives a failing cycle, alerts on failure."""

import core.scheduler as scheduler_mod
from core.scheduler import run_interval


class FakeAlerter:
    def __init__(self):
        self.messages = []

    async def send_text(self, text, chat_id=None):
        self.messages.append(text)


async def test_runs_max_cycles(monkeypatch):
    async def fake_sleep(_):
        return None

    monkeypatch.setattr(scheduler_mod.asyncio, "sleep", fake_sleep)
    count = {"n": 0}

    async def work():
        count["n"] += 1

    await run_interval(work, 0.01, max_cycles=3)
    assert count["n"] == 3


async def test_failing_cycle_does_not_stop_loop(monkeypatch):
    async def fake_sleep(_):
        return None

    monkeypatch.setattr(scheduler_mod.asyncio, "sleep", fake_sleep)
    alerter = FakeAlerter()
    runs = {"n": 0}

    async def work():
        runs["n"] += 1
        if runs["n"] == 2:
            raise RuntimeError("cycle 2 boom")

    await run_interval(work, 0.01, alerter=alerter, label="poll", max_cycles=3)
    # All 3 cycles ran despite cycle 2 failing.
    assert runs["n"] == 3
    assert any("poll failed" in m for m in alerter.messages)


async def test_run_immediately_false_sleeps_first(monkeypatch):
    order = []

    async def fake_sleep(_):
        order.append("sleep")

    monkeypatch.setattr(scheduler_mod.asyncio, "sleep", fake_sleep)

    async def work():
        order.append("work")

    await run_interval(work, 0.01, run_immediately=False, max_cycles=1)
    assert order[0] == "sleep"
    assert "work" in order
