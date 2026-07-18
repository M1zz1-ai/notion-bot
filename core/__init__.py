"""Shared, bot-agnostic core.

Generic building blocks for a Telegram bot backed by an LLM agent:
Telegram I/O (:mod:`core.tg`), an OpenAI tool-calling agent
(:mod:`core.openai_agent`), speech-to-text (:mod:`core.stt`), a resilient
interval scheduler (:mod:`core.scheduler`), config loading (:mod:`core.config`)
and a thin ``notion-cli`` wrapper (:mod:`core.notion`).

Each bot capability is exposed as an agent-compatible callable so a single
agent can register every tool it needs.
"""

__version__ = "0.1.0"
