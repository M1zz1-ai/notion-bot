"""core.tgfmt: Markdown → Telegram-safe HTML.

The load-bearing property is that NOTHING reaches Telegram's HTML parser
unescaped — a reply containing ``&`` or ``<`` used to get the whole message
rejected, which looked like the bot going mute.
"""

from __future__ import annotations

import re

import pytest

from core.tgfmt import to_telegram_html

# Telegram's HTML parse mode accepts exactly these tags (verified against
# aiogram 3.29 HtmlDecoration, generated from the Bot API entity spec).
ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "tg-spoiler", "tg-emoji", "blockquote"}
_TAG_RE = re.compile(r"</?([a-zA-Z][-a-zA-Z0-9]*)")


def _tags(html: str) -> set[str]:
    return set(_TAG_RE.findall(html))


_FULL_TAG_RE = re.compile(r"<(/?)([a-zA-Z][-a-zA-Z0-9]*)(?:\s[^>]*)?>")


def _is_sendable(html: str) -> bool:
    """Would Telegram's HTML parser accept this? Allowed tags, properly nested.

    Interleaved tags (``<b><i>x</b></i>``) make the Bot API reject the WHOLE
    message with "can't parse entities", so this is the property that decides
    whether the user gets their reply at all.
    """
    stack: list[str] = []
    for match in _FULL_TAG_RE.finditer(html):
        closing, name = match.group(1), match.group(2)
        if name not in ALLOWED_TAGS:
            return False
        if closing:
            if not stack or stack.pop() != name:
                return False
        else:
            stack.append(name)
    return not stack


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


def test_bold_italic_underscore_inside_bold_nests_correctly() -> None:
    # The digest stanza style the notion agent is told to write.
    assert to_telegram_html("**_13:00_**") == "<b><i>13:00</i></b>"


def test_triple_stars_nest_correctly_instead_of_interleaving() -> None:
    # Left to the bold+italic passes this produced "<b><i>x</b></i>", which
    # Telegram rejects with "can't parse entities" — the whole message is lost.
    assert to_telegram_html("***13:00***") == "<b><i>13:00</i></b>"


def test_triple_underscores_nest_correctly() -> None:
    assert to_telegram_html("___13:00___") == "<b><i>13:00</i></b>"


def test_digest_style_stanzas_convert_to_the_house_look() -> None:
    reply = (
        "🎯 **_09:00_** *Разбор почты*\n\n"
        "⚡ **_13:00_** *M1zz1 OS & дашборд*\n\n"
        "📅 **Без времени**\n🎯 *Позвонить Саше*\n\n"
        "```\n✅ Done (1)\n • Зарядка\n```\n\n"
        "**Итого: 4 задачи**"
    )
    out = to_telegram_html(reply)
    assert _tags(out) <= ALLOWED_TAGS
    assert "🎯 <b><i>09:00</i></b> <i>Разбор почты</i>" in out
    assert "⚡ <b><i>13:00</i></b> <i>M1zz1 OS &amp; дашборд</i>" in out
    assert "📅 <b>Без времени</b>" in out
    assert "<pre>✅ Done (1)\n • Зарядка\n</pre>" in out
    assert out.endswith("<b>Итого: 4 задачи</b>")


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


# ---- nesting fallback (the other crash) ---------------------------------
#
# Telegram rejects a message whose tags interleave, exactly as it rejects an
# unescaped `<`, and with the same symptom: the entire reply is dropped. The
# converter therefore checks its own output and, when the nesting is broken,
# returns the plain escaped text instead. Ugly beats lost.

# Notion task titles carrying these markers are interpolated into the digest
# stanza the notion agent is prompted to write; each combination below produced
# interleaved HTML before the fallback existed.
_BREAKING_TITLES = ["**bold** start", "***triple***", "****quad****"]
_STANZA_TEMPLATES = [
    "🎯 **_09:00_** *{t}*",
    "🎯 *{t}*",
    "📅 **Без времени**\n🎯 *{t}*",
    "**{t}**",
    "🎯 **_09:00_** *{t}*\n\n**Итого: 1 задача**",
]


@pytest.mark.parametrize("title", _BREAKING_TITLES)
@pytest.mark.parametrize("template", _STANZA_TEMPLATES)
def test_adversarial_task_titles_stay_sendable(title: str, template: str) -> None:
    assert _is_sendable(to_telegram_html(template.format(t=title)))


def test_interleaved_output_falls_back_to_escaped_original() -> None:
    # `**bold** start` inside the stanza's `*…*` gives `<b><i>bold</b> start</i>`.
    out = to_telegram_html("🎯 ***bold** start*")
    assert out == "🎯 ***bold** start*"
    assert _tags(out) == set()


def test_fallback_still_escapes_the_metacharacters() -> None:
    # The fallback is the LAST line of defence, so it may not undo step 2:
    # a raw `&` or `<` in the plain text would fail the send just as hard.
    out = to_telegram_html("***a & b** <c> start*")
    assert out == "***a &amp; b** &lt;c&gt; start*"


def test_four_star_runs_are_sendable() -> None:
    # `****x****` used to render `<b><i><b>x</i></b></b>` — interleaved.
    assert _is_sendable(to_telegram_html("****x****"))


def test_wellformed_replies_are_not_flattened_by_the_fallback() -> None:
    # The check must be inert on everything that was already fine.
    reply = "🎯 **_09:00_** *Разбор почты* & `code`\n\n> quote\n\n**Итого: 1 задача**"
    out = to_telegram_html(reply)
    assert _is_sendable(out)
    assert "<b><i>09:00</i></b>" in out
    assert "<i>Разбор почты</i>" in out
