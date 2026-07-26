"""core.tgfmt: Markdown → Telegram-safe HTML.

The load-bearing property is that NOTHING reaches Telegram's HTML parser
unescaped — a reply containing ``&`` or ``<`` used to get the whole message
rejected, which looked like the bot going mute.
"""

from __future__ import annotations

import re

from core.tgfmt import to_telegram_html

# Telegram's HTML parse mode accepts exactly these tags (verified against
# aiogram 3.29 HtmlDecoration, generated from the Bot API entity spec).
ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "tg-spoiler", "tg-emoji", "blockquote"}
_TAG_RE = re.compile(r"</?([a-zA-Z][-a-zA-Z0-9]*)")


def _tags(html: str) -> set[str]:
    return set(_TAG_RE.findall(html))


# ---- escaping (the crash) ----------------------------------------------


def test_ampersand_is_escaped() -> None:
    # "Schedule & Secretary" is a real project name; unescaped it kills the send.
    assert to_telegram_html("Schedule & Secretary") == "Schedule &amp; Secretary"


def test_angle_brackets_are_escaped() -> None:
    assert to_telegram_html("a < b > c") == "a &lt; b &gt; c"


def test_model_emitted_html_is_neutralised_not_executed() -> None:
    # If the model ignores the prompt and writes HTML, it must render as text
    # rather than smuggling a tag Telegram would reject.
    assert to_telegram_html("<script>x</script>") == "&lt;script&gt;x&lt;/script&gt;"


def test_bold_and_ampersand_together() -> None:
    """The exact DoD case: & , < and **bold** in one reply."""
    out = to_telegram_html("**Schedule & Secretary** uses a < b")
    assert out == "<b>Schedule &amp; Secretary</b> uses a &lt; b"
    assert _tags(out) <= ALLOWED_TAGS


# ---- inline conversion --------------------------------------------------


def test_bold_asterisks_become_b_tag() -> None:
    assert to_telegram_html("**done**") == "<b>done</b>"


def test_italic_and_strike() -> None:
    assert to_telegram_html("*soon* and ~~gone~~") == "<i>soon</i> and <s>gone</s>"


def test_underscores_inside_a_word_are_not_italics() -> None:
    # Page ids and snake_case identifiers must survive intact.
    assert to_telegram_html("call check_habit now") == "call check_habit now"


def test_inline_code_contents_are_escaped_and_not_interpreted() -> None:
    assert to_telegram_html("`a & *b*`") == "<code>a &amp; *b*</code>"


def test_link_renders_as_anchor() -> None:
    assert to_telegram_html("[docs](https://x.io/a)") == '<a href="https://x.io/a">docs</a>'


def test_link_href_is_not_touched_by_emphasis() -> None:
    out = to_telegram_html("[t](https://x.io/a_b_c/*d*)")
    assert "<i>" not in out
    assert 'href="https://x.io/a_b_c/*d*"' in out


# ---- block structure ----------------------------------------------------


def test_heading_becomes_bold_line() -> None:
    # Telegram has no <h1>; a heading tag would be rejected outright.
    assert to_telegram_html("## Today") == "<b>Today</b>"


def test_bullets_become_bullet_characters() -> None:
    # Telegram has no <ul>/<li> either.
    out = to_telegram_html("- gym\n- trading")
    assert out == "• gym\n• trading"
    assert _tags(out) <= ALLOWED_TAGS


def test_consecutive_quote_lines_collapse_into_one_blockquote() -> None:
    assert to_telegram_html("> one\n> two") == "<blockquote>one\ntwo</blockquote>"


def test_fenced_code_becomes_pre_with_escaped_body() -> None:
    assert to_telegram_html("```\na & b\n```") == "<pre>a &amp; b\n</pre>"


def test_full_reply_uses_only_telegram_tags() -> None:
    reply = (
        "## План\n- **Gym** в 17:00\n- Разобрать `pending.json` & отчёт\n\n> перенёс с 23:30\n*всё*"
    )
    assert _tags(to_telegram_html(reply)) <= ALLOWED_TAGS


def test_empty_input_is_empty_output() -> None:
    assert to_telegram_html("") == ""
