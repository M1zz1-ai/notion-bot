"""Config loading: per-bot required keys, defaults, fail-loud on missing."""

import pytest

from core.config import Config, load
from core.errors import ConfigError

ENV = """\
TELEGRAM_BOT_TOKEN=tok
TELEGRAM_CHAT_ID=100000001
ANTHROPIC_API_KEY=sk-test
FAL_KEY=fal-xyz
"""


def _write(tmp_path, content=ENV):
    p = tmp_path / ".env"
    p.write_text(content)
    return p


def test_load_only_requested_keys(tmp_path):
    cfg = load(["TELEGRAM_BOT_TOKEN", "FAL_KEY"], env_path=_write(tmp_path))
    assert isinstance(cfg, Config)
    assert cfg.require("TELEGRAM_BOT_TOKEN") == "tok"
    assert cfg.require("FAL_KEY") == "fal-xyz"
    # A key the bot did not request is simply absent.
    assert cfg.get("ANTHROPIC_API_KEY") is None


def test_missing_key_raises_named_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load(["TELEGRAM_BOT_TOKEN", "GITHUB_TOKEN"], env_path=_write(tmp_path))
    assert "GITHUB_TOKEN" in str(exc.value)


def test_empty_value_treated_as_missing(tmp_path):
    env = ENV.replace("FAL_KEY=fal-xyz", "FAL_KEY=")
    with pytest.raises(ConfigError) as exc:
        load(["FAL_KEY"], env_path=_write(tmp_path, env))
    assert "FAL_KEY" in str(exc.value)


def test_redis_url_default_when_absent(tmp_path):
    cfg = load(["REDIS_URL"], env_path=_write(tmp_path))
    assert cfg.require("REDIS_URL") == "redis://localhost:6379"


def test_redis_url_override_from_env(tmp_path):
    env = ENV + "REDIS_URL=redis://otherhost:6380\n"
    cfg = load(["REDIS_URL"], env_path=_write(tmp_path, env))
    assert cfg.require("REDIS_URL") == "redis://otherhost:6380"


def test_attribute_access(tmp_path):
    cfg = load(["FAL_KEY"], env_path=_write(tmp_path))
    assert cfg.fal_key == "fal-xyz"
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent_key


def test_per_bot_keys_are_independent(tmp_path):
    p = _write(tmp_path)
    image_cfg = load(["TELEGRAM_BOT_TOKEN", "FAL_KEY"], env_path=p)
    chat_cfg = load(["ANTHROPIC_API_KEY"], env_path=p)
    assert image_cfg.get("ANTHROPIC_API_KEY") is None
    assert chat_cfg.get("FAL_KEY") is None
