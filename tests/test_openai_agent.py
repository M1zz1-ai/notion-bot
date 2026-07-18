"""OpenAI agent: plain + structured completions against a fake client (no network).

Mirrors the SDK response shape the code reads: ``response.choices[0].message.content``
and ``response.usage.prompt_tokens`` / ``completion_tokens``.

No ``from __future__ import annotations`` here, on purpose: the tool-registration
test needs real type objects (not PEP-563 strings) so ``_schema_from_signature``
resolves ``int`` -> ``"integer"`` — same as ``tests/test_agent.py``.
"""

from dataclasses import dataclass

import pytest

from core.errors import AgentError
from core.openai_agent import DEFAULT_MODEL, OpenAIAgent

# ---- fake SDK objects --------------------------------------------------


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Fn
    type: str = "function"


@dataclass
class _Msg:
    content: str | None
    tool_calls: list | None = None


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Usage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class _Response:
    content: str | None
    usage: _Usage = None  # type: ignore[assignment]
    tool_calls: list | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _Usage()

    @property
    def choices(self) -> list[_Choice]:
        return [_Choice(_Msg(self.content, tool_calls=self.tool_calls))]


class _FakeCompletions:
    """Returns scripted responses in order; records every create() kwargs."""

    def __init__(self, responses, raise_exc=None):
        self._responses = list(responses)
        self._raise = raise_exc
        self.calls: list[dict] = []

    def create(self, **kwargs):
        recorded = dict(kwargs)
        if "messages" in recorded:
            recorded["messages"] = list(recorded["messages"])
        self.calls.append(recorded)
        if self._raise is not None:
            raise self._raise
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, responses, raise_exc=None):
        self.chat = _FakeChat(_FakeCompletions(responses, raise_exc=raise_exc))


# ---- tests -------------------------------------------------------------


def test_default_model():
    agent = OpenAIAgent(FakeClient([]))
    assert agent.model == DEFAULT_MODEL == "gpt-5.5"


def test_run_returns_text_and_sends_system():
    client = FakeClient([_Response("the answer")])
    agent = OpenAIAgent(client, system="You are helpful.", model="gpt-5.4-mini", max_tokens=1500)
    assert agent.run("question") == "the answer"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-5.4-mini"
    # reasoning models take max_completion_tokens, never max_tokens
    assert call["max_completion_tokens"] == 1500
    assert "max_tokens" not in call
    assert call["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert call["messages"][-1] == {"role": "user", "content": "question"}


def test_run_no_system_omits_system_message():
    client = FakeClient([_Response("hi")])
    agent = OpenAIAgent(client)
    agent.run("q")
    roles = [m["role"] for m in client.chat.completions.calls[0]["messages"]]
    assert roles == ["user"]


def test_run_none_content_becomes_empty_string():
    agent = OpenAIAgent(FakeClient([_Response(None)]))
    assert agent.run("q") == ""


def test_usage_is_surfaced():
    client = FakeClient([_Response("hi", usage=_Usage(prompt_tokens=100, completion_tokens=20))])
    agent = OpenAIAgent(client)
    agent.run("q")
    assert agent.last_usage == {"input_tokens": 100, "output_tokens": 20}


def test_run_api_failure_raises_agent_error():
    client = FakeClient([], raise_exc=RuntimeError("openai down"))
    agent = OpenAIAgent(client)
    with pytest.raises(AgentError):
        agent.run("q")


def test_history_persists_across_runs_when_enabled():
    client = FakeClient([_Response("first"), _Response("second")])
    agent = OpenAIAgent(client, keep_history=True)
    agent.run("a")
    agent.run("b")
    second = client.chat.completions.calls[1]["messages"]
    # [user a, assistant first, user b]
    assert second[0] == {"role": "user", "content": "a"}
    assert second[1] == {"role": "assistant", "content": "first"}
    assert second[-1] == {"role": "user", "content": "b"}


def test_structured_output_parses_json_and_sends_schema():
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    client = FakeClient([_Response('{"label": "spam"}')])
    agent = OpenAIAgent(client)
    out = agent.structured_output("classify this", schema)
    assert out == {"label": "spam"}
    fmt = client.chat.completions.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == schema


def test_structured_output_invalid_json_raises():
    agent = OpenAIAgent(FakeClient([_Response("not json")]))
    with pytest.raises(AgentError):
        agent.structured_output("x", {"type": "object"})


def test_structured_output_api_failure_raises():
    client = FakeClient([], raise_exc=RuntimeError("boom"))
    agent = OpenAIAgent(client)
    with pytest.raises(AgentError):
        agent.structured_output("x", {"type": "object"})


# ---- tool loop ---------------------------------------------------------


def test_tool_registration_infers_schema():
    agent = OpenAIAgent(FakeClient([]))

    @agent.tool
    def add(a: int, b: int = 0) -> int:
        """Add two numbers."""
        return a + b

    params = agent._tool_params()
    assert params[0]["type"] == "function"
    fn = params[0]["function"]
    assert fn["name"] == "add"
    assert fn["description"] == "Add two numbers."
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"
    assert fn["parameters"]["required"] == ["a"]  # b has a default


def test_no_tools_omits_tools_param():
    client = FakeClient([_Response("hi")])
    OpenAIAgent(client).run("q")
    assert "tools" not in client.chat.completions.calls[0]


def test_tool_loop_executes_and_feeds_result_back():
    client = FakeClient(
        [
            _Response(None, tool_calls=[_ToolCall("t1", _Fn("add", '{"a": 2, "b": 3}'))]),
            _Response("result is 5"),
        ]
    )
    agent = OpenAIAgent(client)
    calls = {}

    @agent.tool
    def add(a: int, b: int) -> int:
        """Add."""
        calls["args"] = (a, b)
        return a + b

    assert agent.run("add 2 and 3") == "result is 5"
    assert calls["args"] == (2, 3)
    # Second call carries the assistant tool_calls turn + the tool result message.
    second = client.chat.completions.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "t1"
    assert second[-1] == {"role": "tool", "tool_call_id": "t1", "content": "5"}


def test_tool_error_surfaced_not_crashed():
    client = FakeClient(
        [
            _Response(None, tool_calls=[_ToolCall("t1", _Fn("boom", "{}"))]),
            _Response("handled"),
        ]
    )
    agent = OpenAIAgent(client)

    @agent.tool
    def boom() -> str:
        """Boom."""
        raise ValueError("kaboom")

    assert agent.run("go") == "handled"
    result = client.chat.completions.calls[1]["messages"][-1]["content"]
    assert "Error" in result and "kaboom" in result


def test_unknown_tool_surfaced():
    client = FakeClient(
        [
            _Response(None, tool_calls=[_ToolCall("t1", _Fn("ghost", "{}"))]),
            _Response("ok"),
        ]
    )
    agent = OpenAIAgent(client)  # no tools registered -> ghost is unknown
    agent.run("go")
    assert "unknown tool ghost" in client.chat.completions.calls[1]["messages"][-1]["content"]


def test_tool_loop_iteration_cap_raises():
    forever = [
        _Response(None, tool_calls=[_ToolCall("t", _Fn("noop", "{}"))]) for _ in range(20)
    ]
    client = FakeClient(forever)
    agent = OpenAIAgent(client)

    @agent.tool
    def noop() -> str:
        """Noop."""
        return "ok"

    with pytest.raises(AgentError):
        agent.run("loop", max_iterations=3)
