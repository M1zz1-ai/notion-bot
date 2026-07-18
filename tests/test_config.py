"""Config loading: per-consumer required keys, defaults, fail-loud on missing."""

import pytest

from core.config import Config, load
from core.errors import ConfigError

ENV = """\
TELEGRAM_BOT_TOKEN_NOTION=tok
TELEGRAM_CHAT_ID=000000000
OPENAI_API_KEY=oa-test
EXTRA_KEY=extra-xyz
"""


def _write(tmp_path, content=ENV):
    p = tmp_path / ".env"
    p.write_text(content)
    return p


def test_load_only_requested_keys(tmp_path):
    cfg = load(["TELEGRAM_BOT_TOKEN_NOTION", "EXTRA_KEY"], env_path=_write(tmp_path))
    assert isinstance(cfg, Config)
    assert cfg.require("TELEGRAM_BOT_TOKEN_NOTION") == "tok"
    assert cfg.require("EXTRA_KEY") == "extra-xyz"
    # A key the bot did not request is simply absent.
    assert cfg.get("OPENAI_API_KEY") is None


def test_missing_key_raises_named_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load(["TELEGRAM_BOT_TOKEN_NOTION", "GITHUB_TOKEN"], env_path=_write(tmp_path))
    assert "GITHUB_TOKEN" in str(exc.value)


def test_empty_value_treated_as_missing(tmp_path):
    env = ENV.replace("EXTRA_KEY=extra-xyz", "EXTRA_KEY=")
    with pytest.raises(ConfigError) as exc:
        load(["EXTRA_KEY"], env_path=_write(tmp_path, env))
    assert "EXTRA_KEY" in str(exc.value)


def test_attribute_access(tmp_path):
    cfg = load(["EXTRA_KEY"], env_path=_write(tmp_path))
    assert cfg.extra_key == "extra-xyz"
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent_key


def test_per_consumer_keys_are_independent(tmp_path):
    p = _write(tmp_path)
    a_cfg = load(["TELEGRAM_BOT_TOKEN_NOTION", "EXTRA_KEY"], env_path=p)
    b_cfg = load(["OPENAI_API_KEY"], env_path=p)
    assert a_cfg.get("OPENAI_API_KEY") is None
    assert b_cfg.get("EXTRA_KEY") is None
