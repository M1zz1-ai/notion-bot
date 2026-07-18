"""Telegram layer: pure helpers (chunking, callback parse, keyboards) + client
send paths with a fake aiogram Bot (no network)."""

import pytest

from core.tg import (
    Callback,
    TelegramClient,
    approval_keyboard,
    chunk_text,
    gather_chat_ids,
    inline_keyboard,
    parse_callback,
    reply_keyboard,
)

# ---- chunking ----------------------------------------------------------


def test_chunk_short_text_single_chunk():
    assert chunk_text("hello") == ["hello"]


def test_chunk_empty_text():
    assert chunk_text("") == []


def test_chunk_splits_past_limit_on_newline():
    text = "a" * 30 + "\n" + "b" * 30
    chunks = chunk_text(text, limit=40)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 30
    assert chunks[1] == "b" * 30


def test_chunk_hard_splits_a_long_unbroken_line():
    text = "x" * 100
    chunks = chunk_text(text, limit=40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks) == text


# ---- callback parsing --------------------------------------------------


def test_parse_callback_basic():
    cb = parse_callback("gmail:read:msg123")
    assert cb == Callback("gmail", "read", "msg123", (), "gmail:read:msg123")


def test_parse_callback_arg_keeps_colons():
    cb = parse_callback("approve:yes:tok42:more")
    assert cb.namespace == "approve"
    assert cb.action == "yes"
    assert cb.arg == "tok42:more"
    assert cb.extra == ()


@pytest.mark.parametrize("value", ["16:9", "09:00", "tok123"])
def test_parse_callback_round_trips_colon_values(value):
    # Inverse of how keyboards build callback_data: ns:action:value.
    data = f"img:aspect:{value}"
    cb = parse_callback(data)
    assert cb is not None
    assert cb.namespace == "img"
    assert cb.action == "aspect"
    assert cb.arg == value
    assert cb.raw == data


def test_inline_keyboard_callback_round_trips_via_parse():
    kb = inline_keyboard([[("16:9", "img:aspect:16:9")]])
    built = kb.inline_keyboard[0][0].callback_data
    cb = parse_callback(built)
    assert cb is not None
    assert (cb.namespace, cb.action, cb.arg) == ("img", "aspect", "16:9")


@pytest.mark.parametrize("bad", [None, "", "only", "ns:action", "ns::arg", ":action:arg"])
def test_parse_callback_rejects_malformed(bad):
    assert parse_callback(bad) is None


# ---- keyboards ---------------------------------------------------------


def test_inline_keyboard_shape():
    kb = inline_keyboard([[("A", "ns:a:1"), ("B", "ns:b:2")]])
    assert kb.inline_keyboard[0][0].text == "A"
    assert kb.inline_keyboard[0][1].callback_data == "ns:b:2"


def test_reply_keyboard_is_persistent_and_resized():
    kb = reply_keyboard([["auto", "1:1"], ["📝 Text→Image"]])
    assert kb.is_persistent and kb.resize_keyboard
    assert kb.keyboard[0][0].text == "auto"
    assert kb.keyboard[0][1].text == "1:1"
    assert kb.keyboard[1][0].text == "📝 Text→Image"


def test_approval_keyboard_callbacks():
    kb = approval_keyboard("tok99", prefix="img")
    buttons = kb.inline_keyboard[0]
    assert buttons[0].callback_data == "img:yes:tok99"
    assert buttons[1].callback_data == "img:no:tok99"


def test_gather_chat_ids():
    assert gather_chat_ids("123, 456 789") == [123, 456, 789]


# ---- client send paths (fake Bot) --------------------------------------


class FakeBot:
    def __init__(self):
        self.sent = []
        self.photos = []
        self.session = type("S", (), {"close": _noop})()

    async def send_message(self, chat_id, text, reply_markup=None, disable_web_page_preview=True):
        self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
        return {"message_id": len(self.sent)}

    async def send_photo(self, chat_id, photo, caption=None, reply_markup=None):
        self.photos.append({"chat_id": chat_id, "photo": photo, "caption": caption})
        return {"message_id": 1}


async def _noop():
    return None


def _client(default_chat=None):
    return TelegramClient(FakeBot(), default_chat_id=default_chat)


async def test_send_text_uses_default_chat():
    c = _client(default_chat=777)
    await c.send_text("hi")
    assert c.bot.sent[0]["chat_id"] == 777


async def test_send_text_requires_a_chat():
    c = _client()  # no default
    with pytest.raises(ValueError):
        await c.send_text("hi")


async def test_send_text_chunks_long_message_keyboard_on_last():
    c = _client(default_chat=1)
    long = ("x" * 4096) + "\n" + ("y" * 100)
    kb = inline_keyboard([[("ok", "ns:ok:1")]])
    sent = await c.send_text(long, reply_markup=kb)
    assert len(sent) == 2
    assert c.bot.sent[0]["reply_markup"] is None
    assert c.bot.sent[1]["reply_markup"] is kb


async def test_send_photo_url():
    c = _client(default_chat=5)
    await c.send_photo("https://x/y.png", caption="cat")
    assert c.bot.photos[0]["photo"] == "https://x/y.png"
    assert c.bot.photos[0]["caption"] == "cat"


async def test_request_approval_attaches_keyboard():
    c = _client(default_chat=9)
    msg = await c.request_approval("Approve this?", "tok7", prefix="job")
    assert msg["message_id"] == 1
    kb = c.bot.sent[-1]["reply_markup"]
    assert kb.inline_keyboard[0][0].callback_data == "job:yes:tok7"
