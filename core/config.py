"""Configuration loading from the single master env file.

All secrets/config come from ~/.config/m1zz1/.env (same path as gmail-bot-py).
Each bot declares the keys IT needs via ``load(required=[...])`` — the core
never hard-requires a fixed set, so a new bot adds new keys without touching
this module.

Required keys fail loud (ConfigError naming the missing key) so a misconfigured
deploy never silently runs with empty credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from .errors import ConfigError

LOCAL_ENV_PATH = Path(".env")
MASTER_ENV_PATH = (
    LOCAL_ENV_PATH if LOCAL_ENV_PATH.exists() else Path.home() / ".config" / "m1zz1" / ".env"
)

# Defaults for optional keys. A bot may still declare REDIS_URL as required;
# if absent from the env file it falls back to this rather than raising.
DEFAULTS: dict[str, str] = {
    "REDIS_URL": "redis://localhost:6379",
    # Wall-clock scheduling (notion-bot digest + slot pinger). Dublin is where
    # Bogdan actually is; it is +01:00/+00:00 with DST, which is exactly why the
    # zone stays a config key (a ZoneInfo name) and never a baked-in offset.
    # Historical Notion rows still carry +03:00 — those are legacy offsets on
    # the data, not a statement about where the clock lives.
    "TIMEZONE": "Europe/Dublin",
    "DIGEST_HOUR": "11",
}


@dataclass(frozen=True)
class Config:
    """Loaded config values. Access via attribute (``cfg.anthropic_api_key``)
    or by raw env key (``cfg.get("TELEGRAM_BOT_TOKEN")``).

    Only the keys a bot requested (plus any present defaults) are populated.
    """

    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Return the value for an env key, or ``default`` if absent."""
        return self.values.get(key, default)

    def require(self, key: str) -> str:
        """Return the value for an env key or raise ConfigError if absent."""
        value = self.values.get(key)
        if value is None:
            raise ConfigError(f"Missing required config key: {key}")
        return value

    def __getattr__(self, name: str) -> str:
        """Attribute access by lower-snake env key (e.g. ``cfg.fal_key``)."""
        key = name.upper()
        value = self.values.get(key)
        if value is None:
            raise AttributeError(name)
        return value


def load(required: list[str], *, env_path: Path = MASTER_ENV_PATH) -> Config:
    """Load and validate the env file for one bot's declared keys.

    Does not mutate ``os.environ`` (uses ``dotenv_values``), so loading config
    for a ``--check`` never leaks secrets into the process environment.

    A required key is satisfied by a non-empty value in the env file, or by an
    entry in :data:`DEFAULTS` (e.g. ``REDIS_URL``).

    Args:
        required: Env keys this bot needs (e.g. ["TELEGRAM_BOT_TOKEN", "FAL_KEY"]).
        env_path: Override for the master env file (tests pass a tmp path).

    Raises:
        ConfigError: if any required key is absent/empty and has no default.
    """
    raw = dict(dotenv_values(env_path))
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for key in required:
        value = raw.get(key)
        if value is not None and value.strip() != "":
            resolved[key] = value.strip()
        elif key in DEFAULTS:
            resolved[key] = DEFAULTS[key]
        else:
            missing.append(key)

    if missing:
        raise ConfigError(f"Missing required config key(s): {', '.join(missing)}")

    return Config(values=resolved)
