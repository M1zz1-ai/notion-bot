"""notion.timeslots: the ONE place a Notion date string becomes a moment.

Notion returns three shapes on the same property: an offset-aware datetime
(historical rows carry +03:00), a naive datetime, and a date-only string. Each
means something different to the pinger, so each is pinned here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from notion import timeslots

DUBLIN = ZoneInfo("Europe/Dublin")


def test_aware_value_is_used_as_is() -> None:
    """A legacy +03:00 row is a moment, not a wall clock: 12:00 UTC, not 14:00."""
    slot = timeslots.parse_slot("2026-07-26T15:00:00.000+03:00", DUBLIN)
    assert slot == datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def test_aware_value_in_another_offset_is_not_reinterpreted() -> None:
    """+00:00 means UTC — attaching Dublin to it would move the ping by an hour."""
    slot = timeslots.parse_slot("2026-07-26T15:00:00.000+00:00", DUBLIN)
    assert slot == datetime(2026, 7, 26, 15, 0, tzinfo=timezone.utc)


def test_naive_value_gets_the_configured_zone() -> None:
    slot = timeslots.parse_slot("2026-07-26T15:00:00", DUBLIN)
    assert slot == datetime(2026, 7, 26, 14, 0, tzinfo=timezone.utc)  # IST = +01:00


def test_date_only_has_no_time_component() -> None:
    """Date-only tasks belong to the morning digest, not to the pinger."""
    assert timeslots.parse_slot("2026-07-26", DUBLIN) is None


def test_blank_and_garbage_are_none() -> None:
    assert timeslots.parse_slot("", DUBLIN) is None
    assert timeslots.parse_slot(None, DUBLIN) is None
    assert timeslots.parse_slot("not a date", DUBLIN) is None


def test_hhmm_renders_in_the_local_zone() -> None:
    slot = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
    assert timeslots.hhmm(slot, DUBLIN) == "13:00"


def test_slot_key_is_offset_independent() -> None:
    """The same moment written two ways must dedup to one ping key."""
    a = timeslots.parse_slot("2026-07-26T15:00:00.000+03:00", DUBLIN)
    b = timeslots.parse_slot("2026-07-26T12:00:00.000+00:00", DUBLIN)
    assert timeslots.slot_key(a) == timeslots.slot_key(b)
