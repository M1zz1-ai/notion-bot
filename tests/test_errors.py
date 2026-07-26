"""Phoenix resilience: a failing cycle is swallowed + alerted, process survives."""

import asyncio

import pytest

from core.errors import ConfigError, CoreError, resilient, run_resilient


class FakeAlerter:
    def __init__(self):
        self.messages = []

    async def send_text(self, text, chat_id=None):
        self.messages.append(text)


def test_exception_hierarchy():
    assert issubclass(ConfigError, CoreError)
    assert issubclass(CoreError, RuntimeError)


async def test_run_resilient_returns_result_on_success():
    async def work():
        return 42

    assert await run_resilient(work) == 42


async def test_run_resilient_swallows_and_alerts():
    alerter = FakeAlerter()

    async def work():
        raise RuntimeError("boom")

    result = await run_resilient(work, alerter=alerter, label="poll")
    assert result is None
    assert len(alerter.messages) == 1
    assert "poll failed" in alerter.messages[0]
    assert "boom" in alerter.messages[0]


async def test_run_resilient_reraises_cancellation():
    async def work():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_resilient(work)


async def test_alert_failure_does_not_mask_original():
    class BrokenAlerter:
        async def send_text(self, text, chat_id=None):
            raise RuntimeError("telegram down")

    async def work():
        raise ValueError("original")

    # Must not raise — both the work error and the alert error are swallowed.
    assert await run_resilient(work, alerter=BrokenAlerter()) is None


async def test_resilient_decorator():
    alerter = FakeAlerter()

    @resilient(alerter=alerter, label="task")
    async def flaky(x):
        if x < 0:
            raise ValueError("negative")
        return x * 2

    assert await flaky(5) == 10
    assert await flaky(-1) is None
    assert any("task failed" in m for m in alerter.messages)
