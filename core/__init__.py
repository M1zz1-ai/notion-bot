"""m1zz1-bots shared core.

Generic, bot-agnostic building blocks for Bogdan's Python+n8n hybrid bots:
Telegram I/O, an Anthropic tool-calling agent, redis state, fal.ai media,
Notion tasks, spreadsheets, and a resilient interval scheduler.

Each bot is a thin module on this core, and each bot capability is exposed as
an agent-compatible callable (see ``core.agent``) so all bots' tools can later
be registered into one unified agent.
"""

__version__ = "0.1.0"
