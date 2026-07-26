"""Reusable aiogram-based Telegram layer, bot-agnostic.

Generalizes gmail-bot-py's telegram_bot.py: a TelegramClient built from ANY
token (so per-bot tokens like TELEGRAM_BOT_TOKEN_IMAGE coexist), text sending
with auto-chunking past Telegram's 4096-char limit, photo/document sending,
inline keyboards, callback parsing, and a generic human-in-the-loop approval
helper.

Card templates and callback semantics live in each bot, not here — this module
is plumbing only.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

TELEGRAM_MAX_CHARS = 4096


# ---- keyboards ---------------------------------------------------------


def inline_keyboard(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    """Build an inline keyboard from rows of (text, callback_data) tuples.

    Example: ``inline_keyboard([[("Yes", "ok:yes"), ("No", "ok:no")]])``.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in rows
        ]
    )


def reply_keyboard(rows: list[list[str]]) -> ReplyKeyboardMarkup:
    """Build a persistent reply keyboard (docked at the chat's bottom).

    Rows are plain button labels; presses arrive as ordinary text messages
    whose text equals the label. ``resize_keyboard`` shrinks the keyboard to
    fit and ``is_persistent`` keeps it visible after a press.
    """
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=label) for label in row] for row in rows],
        resize_keyboard=True,
        is_persistent=True,
    )


def approval_keyboard(
    token: str, *, approve: str = "✅ Approve", reject: str = "❌ Reject", prefix: str = "approve"
) -> InlineKeyboardMarkup:
    """Two-button approve/reject keyboard for a human-in-the-loop decision.

    Callback data is ``<prefix>:yes:<token>`` / ``<prefix>:no:<token>``; parse
    it with :func:`parse_callback` and match on ``token`` to your pending item.
    """
    return inline_keyboard(
        [
            [
                (approve, f"{prefix}:yes:{token}"),
                (reject, f"{prefix}:no:{token}"),
            ]
        ]
    )


# ---- callback parsing --------------------------------------------------


@dataclass(frozen=True)
class Callback:
    """A parsed ``<namespace>:<action>:<arg>`` callback (``arg`` may contain colons)."""

    namespace: str
    action: str
    arg: str
    extra: tuple[str, ...]
    raw: str


def parse_callback(data: str | None) -> Callback | None:
    """Parse ``ns:action:arg``. Returns None for empty/malformed.

    Only the first two colons are split, so ``arg`` keeps any colons it
    contains (e.g. aspect ratio ``16:9`` or time ``09:00``) — symmetric with how
    keyboards build ``<ns>:<action>:<value>`` callback_data. ``extra`` stays for
    backward compat and is always empty. The bot decides what each segment means;
    this only splits and validates the minimum 3-part shape.
    """
    if not data:
        return None
    parts = data.split(":", 2)
    if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
        return None
    return Callback(
        namespace=parts[0],
        action=parts[1],
        arg=parts[2],
        extra=(),
        raw=data,
    )


# ---- text chunking -----------------------------------------------------


def chunk_text(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split text into <=limit-char chunks, preferring newline boundaries.

    Telegram rejects messages over 4096 chars; long agent output must be split.
    A single oversized line (no newline within ``limit``) is hard-split.

    The split is TAG-AWARE, because the client sends with ``parse_mode="HTML"``
    and every chunk is parsed as a standalone document. A naive character split
    breaks a long formatted reply two ways: it can land inside ``<blockquote>``
    (both halves become garbage) or between an open tag and its close (the first
    chunk has an unclosed tag, the second a stray closing one) — and Telegram
    rejects the whole message either way. So a cut never falls inside a tag, any
    tags still open at the cut are closed at the end of the chunk, and reopened
    at the start of the next. Text with no tags takes exactly the old path.
    """
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut, open_tags = _find_cut(remaining, limit)
        closing = "".join(f"</{name}>" for name, _ in reversed(open_tags))
        reopening = "".join(f"<{name}{attrs}>" for name, attrs in open_tags)
        chunks.append(remaining[:cut] + closing)
        remaining = reopening + remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


_TAG_RE = re.compile(r"<(/?)([a-zA-Z][-a-zA-Z0-9]*)([^<>]*)>")


def _find_cut(text: str, limit: int) -> tuple[int, list[tuple[str, str]]]:
    """Return (cut index, tags open there) for the best split at or under ``limit``.

    Walks the text once, tracking the open-tag stack. Only offsets that sit
    between tags are candidates, and a candidate must leave room for the closing
    tags the chunk will need. A newline boundary wins when one is reachable;
    otherwise the last in-budget offset is used (the old hard-split, minus the
    risk of cutting a tag in half).
    """
    stack: list[tuple[str, str]] = []
    best = (0, ())  # type: tuple[int, tuple[tuple[str, str], ...]]
    best_newline: tuple[int, tuple[tuple[str, str], ...]] | None = None
    i = 0
    while i < len(text):
        closing_cost = sum(len(name) + 3 for name, _ in stack)
        if i > 0 and i + closing_cost <= limit:
            best = (i, tuple(stack))
            if text[i - 1] == "\n" and i - 1 > 0:
                # Cut BEFORE the newline: the separator is consumed by the split,
                # not carried into the chunk (the newline itself opens/closes
                # nothing, so the tag stack is the same either side of it).
                best_newline = (i - 1, tuple(stack))
        elif i > limit:
            break

        match = _TAG_RE.match(text, i) if text[i] == "<" else None
        if match is None:
            i += 1
            continue
        is_close, name, attrs = match.group(1), match.group(2), match.group(3)
        if is_close:
            # Drop the matching open tag. An unbalanced close (nothing open, or
            # a different tag) is left alone rather than guessed at.
            if stack and stack[-1][0] == name:
                stack.pop()
        elif not attrs.rstrip().endswith("/"):
            stack.append((name, attrs))
        i = match.end()

    cut, tags = best_newline or best
    if cut == 0:
        # Nothing splittable in budget (e.g. one enormous tag): fall back to a
        # flat cut so the loop still terminates.
        return limit, []
    return cut, list(tags)


# ---- client ------------------------------------------------------------


class TelegramClient:
    """Thin wrapper over an aiogram Bot, built from any token.

    Holds an optional ``default_chat_id`` so callers can send without repeating
    it; pass ``chat_id=`` to override per call.
    """

    def __init__(self, bot: Bot, default_chat_id: int | None = None) -> None:
        self._bot = bot
        self._default_chat_id = default_chat_id

    @classmethod
    def from_token(
        cls, token: str, default_chat_id: int | None = None, *, parse_mode: str = "HTML"
    ) -> TelegramClient:
        """Construct a client from a raw bot token (per-bot tokens supported)."""
        bot = Bot(token=token, default=DefaultBotProperties(parse_mode=parse_mode))
        return cls(bot, default_chat_id)

    @property
    def bot(self) -> Bot:
        """The underlying aiogram Bot (for wiring a Dispatcher)."""
        return self._bot

    def _resolve_chat(self, chat_id: int | None) -> int:
        resolved = chat_id if chat_id is not None else self._default_chat_id
        if resolved is None:
            raise ValueError("no chat_id provided and no default_chat_id set")
        return resolved

    async def send_text(
        self,
        text: str,
        chat_id: int | None = None,
        *,
        reply_markup: InlineKeyboardMarkup | ReplyKeyboardMarkup | None = None,
        disable_web_page_preview: bool = True,
    ) -> list[Message]:
        """Send text, auto-chunking past 4096 chars.

        Returns the sent Message(s). The keyboard is attached to the LAST chunk
        only (Telegram shows one keyboard per message; the action belongs at the
        end of a long body).
        """
        chat = self._resolve_chat(chat_id)
        chunks = chunk_text(text)
        sent: list[Message] = []
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            sent.append(
                await self._bot.send_message(
                    chat_id=chat,
                    text=chunk,
                    reply_markup=reply_markup if is_last else None,
                    disable_web_page_preview=disable_web_page_preview,
                )
            )
        return sent

    async def send_photo(
        self,
        photo: str | bytes,
        chat_id: int | None = None,
        *,
        caption: str | None = None,
        reply_markup: InlineKeyboardMarkup | ReplyKeyboardMarkup | None = None,
        filename: str = "image.png",
    ) -> Message:
        """Send a photo by URL/file_id (str) or raw bytes."""
        chat = self._resolve_chat(chat_id)
        media = photo if isinstance(photo, str) else BufferedInputFile(photo, filename=filename)
        return await self._bot.send_photo(
            chat_id=chat, photo=media, caption=caption, reply_markup=reply_markup
        )

    async def send_document(
        self,
        document: str | bytes,
        chat_id: int | None = None,
        *,
        caption: str | None = None,
        filename: str = "file.bin",
    ) -> Message:
        """Send a document by URL/file_id (str) or raw bytes."""
        chat = self._resolve_chat(chat_id)
        media = (
            document
            if isinstance(document, str)
            else BufferedInputFile(document, filename=filename)
        )
        return await self._bot.send_document(chat_id=chat, document=media, caption=caption)

    async def edit_text(
        self,
        message_id: int,
        text: str,
        chat_id: int | None = None,
        *,
        reply_markup: InlineKeyboardMarkup | ReplyKeyboardMarkup | None = None,
    ) -> Message | bool:
        """Edit a previously sent message's text (e.g. flip a card to 'Done')."""
        return await self._bot.edit_message_text(
            chat_id=self._resolve_chat(chat_id),
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
        )

    async def request_approval(
        self,
        text: str,
        token: str,
        chat_id: int | None = None,
        *,
        prefix: str = "approve",
    ) -> Message:
        """Send a human-in-the-loop approval card and return the sent Message.

        The two buttons emit ``<prefix>:yes:<token>`` / ``<prefix>:no:<token>``.
        The bot's callback handler matches ``token`` to its pending item and
        decides; this helper only renders the card. ``send_text`` chunking is
        used, so the keyboard lands on the last chunk.
        """
        sent = await self.send_text(
            text,
            chat_id,
            reply_markup=approval_keyboard(token, prefix=prefix),
        )
        return sent[-1]

    async def close(self) -> None:
        """Close the bot's HTTP session."""
        await self._bot.session.close()


def gather_chat_ids(raw: str) -> list[int]:
    """Parse a comma/whitespace-separated TELEGRAM_CHAT_ID env value to ints."""
    parts = raw.replace(",", " ").split()
    return [int(p) for p in parts]


async def broadcast(client: TelegramClient, text: str, chat_ids: list[int]) -> None:
    """Send the same text to multiple chats concurrently (best-effort each)."""
    await asyncio.gather(*(client.send_text(text, chat_id=cid) for cid in chat_ids))
