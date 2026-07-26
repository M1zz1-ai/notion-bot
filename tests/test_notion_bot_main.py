"""__main__ startup gate: the notion-cli floor is asserted at boot, not at the
first user action that happens to need a missing flag.

Two behaviours are pinned, because they are deliberately different:
  * ``--check`` (the deploy preflight) HARD-fails on a stale CLI;
  * the long-running process starts DEGRADED and shouts to Telegram, because a
    unit that refuses to boot delivers no morning plan and no explanation.
"""

import logging

import pytest

import notion.__main__ as main_mod
import core.notion as notion
from core import config


def _check(ok: bool, *, reason: str = "too_old", found: str | None = "0.1.0"):
    return notion.VersionCheck(
        ok=ok,
        found=found,
        required="0.2.1",
        reason=reason if not ok else "ok",
        message="notion-cli too old: found 0.1.0, requires >= 0.2.1 — fix it",
    )


def test_verify_cli_logs_error_when_stale(monkeypatch, caplog):
    monkeypatch.setattr(notion, "check_cli_version", lambda *a, **k: _check(False))
    with caplog.at_level(logging.ERROR, logger="notion_bot"):
        result = main_mod.verify_cli()
    assert result.ok is False
    assert "notion-cli too old" in caplog.text
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_verify_cli_logs_info_when_ok(monkeypatch, caplog):
    monkeypatch.setattr(notion, "check_cli_version", lambda *a, **k: _check(True, found="0.2.1"))
    with caplog.at_level(logging.INFO, logger="notion_bot"):
        result = main_mod.verify_cli()
    assert result.ok is True
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_check_mode_exits_nonzero_on_stale_cli(monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda keys: config.Config({}))
    monkeypatch.setattr(notion, "check_cli_version", lambda *a, **k: _check(False))
    monkeypatch.setattr("sys.argv", ["notion", "--check"])
    assert main_mod.main() == 2
    assert "notion-cli" in capsys.readouterr().err


def test_check_mode_exits_zero_when_cli_current(monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda keys: config.Config({}))
    monkeypatch.setattr(notion, "check_cli_version", lambda *a, **k: _check(True, found="0.2.1"))
    monkeypatch.setattr("sys.argv", ["notion", "--check"])
    assert main_mod.main() == 0
    assert "notion-cli 0.2.1" in capsys.readouterr().out


class _FakeTg:
    def __init__(self):
        self.sent: list[str] = []

    async def send_text(self, text, chat_id=None, **kw):
        self.sent.append(text)


@pytest.mark.asyncio
async def test_degraded_alert_reaches_telegram():
    tg = _FakeTg()
    await main_mod.alert_if_degraded(tg, _check(False))
    assert len(tg.sent) == 1
    assert "notion-cli too old" in tg.sent[0]


@pytest.mark.asyncio
async def test_no_alert_when_cli_current():
    tg = _FakeTg()
    await main_mod.alert_if_degraded(tg, _check(True, found="0.2.1"))
    assert tg.sent == []


@pytest.mark.asyncio
async def test_alert_failure_does_not_block_startup():
    class Broken:
        async def send_text(self, text, chat_id=None, **kw):
            raise RuntimeError("telegram down")

    await main_mod.alert_if_degraded(Broken(), _check(False))  # must not raise
