"""Markdown → Telegram-safe HTML for LLM replies.

``core.tg.TelegramClient`` sets a bot-wide ``parse_mode="HTML"``, so every
message body is parsed as HTML by Telegram. LLMs emit Markdown by habit, which
produces two distinct bugs:

1. Cosmetic — ``**bold**`` leaks verbatim because asterisks mean nothing in HTML.
2. Fatal — a reply containing ``&`` or ``<`` is rejected by the Bot API with
   ``can't parse entities``, and the send raises. A project named
   "Schedule & Secretary" is enough to do it, and the symptom (no message at
   all) reads as the bot going mute rather than as a formatting bug.

The house approach is CONVERT, not instruct. Telling the model "emit HTML" fixes
(1) but leaves (2) live, because it is then the model's job to remember to write
``&amp;`` inside prose — a rule it will follow most of the time, which is the
worst kind of reliability for a crash. Here escaping is unconditional and
happens before any tag is introduced, so the output cannot carry an unescaped
metacharacter no matter what the model wrote.

Telegram's HTML is NOT a subset of HTML you can guess at — it is a fixed tag
list, verified against aiogram 3.29's entity decorator (``aiogram.utils.
text_decorations.HtmlDecoration``), which is generated from the Bot API entity
spec: ``b i u s code pre a tg-spoiler tg-emoji blockquote``. There is no
``<h1>``, no ``<ul>``/``<li>``, no ``<br>``, no ``<p>``. Headings therefore
render as bold lines and bullets as a literal "•" character — emitting list
tags would be rejected outright. Escaping is ``&``/``<``/``>`` only (quotes are
left alone), matching ``HtmlDecoration.quote``.
"""

from __future__ import annotations

import html
import re

# Sentinel wrapping stashed, already-rendered fragments (code spans, links) so
# the emphasis passes cannot reach inside them. NUL can never survive in model
# output we accept — it is stripped on entry — so it cannot be spoofed.
_SENTINEL = "\x00"
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

_FENCE_RE = re.compile(r"```([^\n`]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+][ \t]+(.*)$")
_QUOTE_RE = re.compile(r"^\s*&gt;[ \t]?(.*)$")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
_BOLD_STAR_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_BOLD_UNDER_RE = re.compile(r"(?<![\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)")
_ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")

BULLET = "•"


def to_telegram_html(text: str) -> str:
    """Render a Markdown-ish LLM reply as Telegram-safe HTML.

    Supported input: ``# heading`` (any level), ``- ``/``* ``/``+ `` bullets,
    ``> `` quotes, ``**bold**``, ``__bold__``, ``*italic*``, ``_italic_``,
    ``~~strike~~``, ``[label](url)``, `````code````` spans and ``` fences.
    Everything else is passed through as escaped literal text.

    The output is safe to send with ``parse_mode="HTML"`` for ANY input,
    including text that already contains ``&``, ``<`` or ``>``.
    """
    if not text:
        return ""

    stash: list[str] = []

    # NULs would collide with the placeholder sentinel; they carry no meaning in
    # a chat reply, so dropping them is lossless in practice and closes the only
    # route by which model output could forge a placeholder.
    text = text.replace(_SENTINEL, "")

    # 1. Code first: its contents must be escaped but NOT interpreted, so it is
    #    lifted out before any Markdown pass can see a `*` or `_` inside it.
    text = _FENCE_RE.sub(
        lambda m: _stash(stash, f"<pre>{html.escape(m.group(2), quote=False)}</pre>"),
        text,
    )
    text = _INLINE_CODE_RE.sub(
        lambda m: _stash(stash, f"<code>{html.escape(m.group(1), quote=False)}</code>"),
        text,
    )

    # 2. Escape EVERYTHING that is left, before a single tag exists. Every `<`
    #    and `&` below this line is one we wrote ourselves.
    text = html.escape(text, quote=False)

    # 3. Block structure, then inline spans within each line.
    rendered = _render_blocks(text, stash)

    # 4. Put the code spans and links back. A stashed anchor can itself contain
    #    a stashed code span (``[`id`](url)``), so resolution repeats until the
    #    text stops changing — bounded by the stash size, which only shrinks as
    #    placeholders are consumed.
    for _ in range(len(stash) + 1):
        expanded = _PLACEHOLDER_RE.sub(lambda m: stash[int(m.group(1))], rendered)
        if expanded == rendered:
            break
        rendered = expanded
    return rendered


def _stash(stash: list[str], rendered: str) -> str:
    """Park an already-rendered fragment behind a placeholder and return it."""
    stash.append(rendered)
    return f"{_SENTINEL}{len(stash) - 1}{_SENTINEL}"


def _render_blocks(text: str, stash: list[str]) -> str:
    """Convert line-oriented Markdown to Telegram HTML (headings/bullets/quotes)."""
    out: list[str] = []
    quote_buf: list[str] = []

    def _flush_quote() -> None:
        if quote_buf:
            out.append("<blockquote>" + "\n".join(quote_buf) + "</blockquote>")
            quote_buf.clear()

    for line in text.split("\n"):
        quoted = _QUOTE_RE.match(line)
        if quoted:
            # Consecutive `> ` lines collapse into ONE blockquote: Telegram
            # renders each blockquote as its own block, so per-line tags would
            # stack a two-line quote into two visually separate quotes.
            quote_buf.append(_render_inline(quoted.group(1), stash))
            continue
        _flush_quote()

        if _RULE_RE.match(line):
            out.append("———")
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            out.append(f"<b>{_render_inline(heading.group(1), stash)}</b>")
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            indent, body = bullet.group(1), bullet.group(2)
            out.append(f"{indent}{BULLET} {_render_inline(body, stash)}")
            continue

        out.append(_render_inline(line, stash))

    _flush_quote()
    return "\n".join(out)


def _render_inline(text: str, stash: list[str]) -> str:
    """Link + emphasis conversion for one line of already-escaped text.

    Links are rendered and stashed FIRST so the emphasis passes below can never
    reach into an href — a url carrying ``*`` or ``__`` would otherwise grow an
    ``<i>`` tag in the middle of the attribute and be rejected by Telegram.
    """
    text = _LINK_RE.sub(lambda m: _render_link(m, stash), text)
    text = _BOLD_STAR_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_UNDER_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    # `_` emphasis requires non-word neighbours so snake_case identifiers and
    # page ids survive intact.
    text = _ITALIC_UNDER_RE.sub(r"<i>\1</i>", text)
    return text


def _render_link(match: re.Match[str], stash: list[str]) -> str:
    """Render ``[label](url)`` as an anchor, escaping quotes inside the href.

    ``&`` and ``<`` in the url are already entities by this point (step 2), but
    ``html.escape(quote=False)`` leaves ``"`` alone — and an unescaped quote in
    an href would terminate the attribute early and produce a tag Telegram
    rejects. The label still gets emphasis (``[**docs**](url)`` is meaningful);
    the href does not, because it is inside the stashed fragment.
    """
    label, href = match.group(1), match.group(2).replace('"', "&quot;")
    return _stash(stash, f'<a href="{href}">{_render_inline(label, stash)}</a>')
