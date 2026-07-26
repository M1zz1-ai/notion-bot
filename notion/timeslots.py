"""Notion date string -> moment. The single timezone seam of the notion bot.

Notion's Date property is returned in three shapes on the very same field:

    "2026-07-26T15:00:00.000+03:00"   offset-aware  -> a real moment, use as is
    "2026-07-26T15:00:00"             naive         -> local wall clock, attach TZ
    "2026-07-26"                      date-only     -> NOT a slot (no time at all)

Getting this wrong is expensive precisely because it looks cheap: a bare
``datetime.now()`` compared against a +03:00 value puts every ping three hours
off and reads as a scheduling bug rather than a timezone bug. So every
comparison downstream happens in UTC, and the only place that decides what a
string means is this module.

A date-only value is deliberately NOT a slot: those tasks belong to the morning
digest, which lists the whole day, not to the pinger, which announces a moment.
"""

from __future__ import annotations

import datetime as dt

__all__ = ["parse_slot", "hhmm", "slot_key"]


def parse_slot(value: str | None, tz: dt.tzinfo) -> dt.datetime | None:
    """Parse a Notion date value into an aware datetime, or None.

    Returns None for an empty value, an unparseable value, or a date-only value
    (no time component). Naive values are interpreted in ``tz``; aware values
    keep the offset Notion sent — reinterpreting them would silently move them.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    # Date-only is decided on the STRING, not on the parsed value: fromisoformat
    # turns "2026-07-26" into midnight, indistinguishable from an explicit 00:00.
    if "T" not in text and " " not in text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed


def hhmm(moment: dt.datetime, tz: dt.tzinfo) -> str:
    """Render a moment as ``HH:MM`` in the local zone (display only)."""
    return moment.astimezone(tz).strftime("%H:%M")


def slot_key(moment: dt.datetime) -> str:
    """Stable dedup key for a slot: the moment normalized to UTC.

    Offset-independent on purpose — the same instant written as +03:00 or as
    +00:00 must produce ONE ping, not two.
    """
    return moment.astimezone(dt.timezone.utc).isoformat()
