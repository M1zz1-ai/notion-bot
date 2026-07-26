"""Redis state: dedup + session via a fake client, plus graceful fallback when
redis raises (the degrade-don't-crash contract)."""

import redis

from core.state import RedisState


class FakeRedis:
    """In-memory stand-in for redis.Redis (no TTL expiry simulated)."""

    def __init__(self):
        self.store = {}
        self.lists = {}

    def ping(self):
        return True

    def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)

    def lrange(self, key, start, stop):
        items = self.lists.get(key, [])
        end = None if stop == -1 else stop + 1
        return items[start:end]

    def ltrim(self, key, start, stop):
        items = self.lists.get(key, [])
        end = None if stop == -1 else stop + 1
        self.lists[key] = items[start:end]

    def exists(self, key):
        return 1 if key in self.store else 0

    def set(self, key, value, ex=None):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)


class DownRedis:
    """Every operation raises, simulating an unreachable server."""

    def ping(self):
        raise redis.ConnectionError("down")

    def exists(self, key):
        raise redis.ConnectionError("down")

    def set(self, key, value, ex=None):
        raise redis.ConnectionError("down")

    def get(self, key):
        raise redis.ConnectionError("down")

    def delete(self, key):
        raise redis.ConnectionError("down")

    def lrange(self, key, start, stop):
        raise redis.ConnectionError("down")

    def ltrim(self, key, start, stop):
        raise redis.ConnectionError("down")


def _state(client):
    return RedisState(client=client, namespace="test")


# ---- dedup -------------------------------------------------------------


def test_dedup_roundtrip():
    s = _state(FakeRedis())
    assert not s.seen("m1")
    s.mark_seen("m1")
    assert s.seen("m1")


def test_dedup_ledgers_are_separate():
    s = _state(FakeRedis())
    s.mark_seen("x", ledger="emails")
    assert s.seen("x", ledger="emails")
    assert not s.seen("x", ledger="news")


def test_namespacing_isolates_bots():
    fake = FakeRedis()
    a = RedisState(client=fake, namespace="bot_a")
    b = RedisState(client=fake, namespace="bot_b")
    a.mark_seen("dup")
    assert a.seen("dup")
    assert not b.seen("dup")


# ---- session -----------------------------------------------------------


def test_session_roundtrip_json():
    s = _state(FakeRedis())
    s.set_session("u1", {"step": 2, "name": "Bogdan"})
    assert s.get_session("u1") == {"step": 2, "name": "Bogdan"}


def test_session_default_on_miss():
    s = _state(FakeRedis())
    assert s.get_session("missing", default="x") == "x"


def test_clear_session():
    s = _state(FakeRedis())
    s.set_session("u1", 1)
    s.clear_session("u1")
    assert s.get_session("u1") is None


# ---- list tail (Mac -> VPS brief) --------------------------------------


def test_last_json_returns_the_newest_element():
    fake = FakeRedis()
    fake.rpush("test:notion:brief", '{"date": "2026-07-25"}', '{"date": "2026-07-26"}')
    assert _state(fake).last_json("notion", "brief") == {"date": "2026-07-26"}


def test_last_json_none_on_empty_list():
    assert _state(FakeRedis()).last_json("notion", "brief") is None


def test_last_json_none_on_garbage():
    fake = FakeRedis()
    fake.rpush("test:notion:brief", "not json")
    assert _state(fake).last_json("notion", "brief") is None


def test_last_json_none_when_down():
    assert _state(DownRedis()).last_json("notion", "brief") is None


def test_trim_list_keeps_only_the_newest():
    """Nothing pops the brief list, so without LTRIM it grows one blob per day."""
    fake = FakeRedis()
    fake.rpush("test:notion:brief", *[f'{{"n": {i}}}' for i in range(10)])
    s = _state(fake)
    s.trim_list("notion", "brief", keep=3)
    assert fake.lists["test:notion:brief"] == ['{"n": 7}', '{"n": 8}', '{"n": 9}']
    assert s.last_json("notion", "brief") == {"n": 9}


def test_trim_list_no_op_when_down():
    _state(DownRedis()).trim_list("notion", "brief", keep=3)  # must not raise


# ---- graceful fallback -------------------------------------------------


def test_ping_false_when_down():
    assert _state(DownRedis()).ping() is False


def test_seen_returns_false_when_down():
    # Down redis -> treat as unseen so the item still gets processed.
    assert _state(DownRedis()).seen("anything") is False


def test_mark_and_set_are_noops_when_down():
    s = _state(DownRedis())
    s.mark_seen("x")  # must not raise
    s.set_session("k", {"v": 1})  # must not raise
    assert s.get_session("k", default=None) is None


def test_degrade_warns_once(caplog):
    s = _state(DownRedis())
    s.seen("a")
    s.seen("b")
    s.mark_seen("c")
    warnings = [r for r in caplog.records if "degrading" in r.message]
    assert len(warnings) == 1
