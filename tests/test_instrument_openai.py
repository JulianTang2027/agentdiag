"""Tests for OpenAI auto-instrumentation (mock-based, no API key needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentdiag.instrument.openai import watch_openai


def _make_response(content=None, tool_calls=None):
    """Build a fake OpenAI ChatCompletion response."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _make_tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _make_client(responses):
    """Build a mock OpenAI client that returns responses in order."""
    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=responses)
    return client


def test_captures_final_answer():
    client = _make_client([
        _make_response(content="The weather is 72F and sunny."),
    ])
    client, tracer = watch_openai(client, task="Get weather")

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather?"}],
    )

    trace = tracer.to_trace()
    assert len(trace.steps) == 1
    assert trace.steps[0].type == "result"
    assert "72F" in trace.steps[0].content


def test_captures_tool_calls():
    client = _make_client([
        _make_response(
            content=None,
            tool_calls=[_make_tool_call("get_weather", '{"city": "Chicago"}')],
        ),
    ])
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    client, tracer = watch_openai(client, task="Get weather")

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Weather in Chicago?"}],
        tools=tools,
    )

    trace = tracer.to_trace()
    assert len(trace.steps) == 1
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "get_weather"
    assert trace.steps[0].tool_args == {"city": "Chicago"}


def test_captures_thought_before_tool_call():
    client = _make_client([
        _make_response(
            content="I should check the weather first.",
            tool_calls=[_make_tool_call("get_weather", '{"city": "NYC"}')],
        ),
    ])
    client, tracer = watch_openai(client, task="Get weather")

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Weather?"}],
    )

    trace = tracer.to_trace()
    assert len(trace.steps) == 2
    assert trace.steps[0].type == "thought"
    assert "check the weather" in trace.steps[0].content
    assert trace.steps[1].type == "tool_call"


def test_captures_observations_from_tool_messages():
    # First call: model makes a tool call
    # Second call: messages include tool result, model gives final answer
    client = _make_client([
        _make_response(
            tool_calls=[_make_tool_call("get_weather", '{"city": "SF"}')],
        ),
        _make_response(content="It's 72F in SF."),
    ])
    client, tracer = watch_openai(client, task="Get weather")

    # First call
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Weather in SF?"}],
    )

    # Second call — includes the tool result
    client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Weather in SF?"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "SF: 72F, sunny", "tool_call_id": "123"},
        ],
    )

    trace = tracer.to_trace()
    types = [s.type for s in trace.steps]
    assert types == ["tool_call", "observation", "result"]
    assert trace.steps[1].content == "SF: 72F, sunny"


def test_auto_extracts_model():
    client = _make_client([_make_response(content="hi")])
    client, tracer = watch_openai(client, task="test")

    client.chat.completions.create(model="gpt-4o-mini", messages=[])

    assert tracer._model == "gpt-4o-mini"


def test_auto_extracts_available_tools():
    tools = [
        {"type": "function", "function": {"name": "search"}},
        {"type": "function", "function": {"name": "write"}},
    ]
    client = _make_client([_make_response(content="done")])
    client, tracer = watch_openai(client, task="test")

    client.chat.completions.create(model="gpt-4o", messages=[], tools=tools)

    assert tracer._available_tools == ["search", "write"]


def test_response_returned_unchanged():
    expected = _make_response(content="hello")
    client = _make_client([expected])
    client, tracer = watch_openai(client, task="test")

    result = client.chat.completions.create(model="gpt-4o", messages=[])

    assert result is expected


def test_error_observation_detected():
    client = _make_client([
        _make_response(
            tool_calls=[_make_tool_call("api_call", '{}')],
        ),
        _make_response(content="I couldn't do it."),
    ])
    client, tracer = watch_openai(client, task="test")

    client.chat.completions.create(model="gpt-4o", messages=[])
    client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "tool", "content": "Error: rate limit exceeded", "tool_call_id": "1"},
        ],
    )

    trace = tracer.to_trace()
    obs = [s for s in trace.steps if s.type == "observation"]
    assert len(obs) == 1
    assert obs[0].error is True


def test_full_agent_loop():
    """Simulate a complete 2-turn agent: tool call → tool result → final answer."""
    client = _make_client([
        # Turn 1: model calls a tool
        _make_response(
            content="Let me search for that.",
            tool_calls=[_make_tool_call("web_search", '{"query": "Python"}')],
        ),
        # Turn 2: model gives final answer
        _make_response(content="Python is a programming language."),
    ])
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    client, tracer = watch_openai(client, task="What is Python?")

    # Turn 1
    client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "user", "content": "What is Python?"},
    ], tools=tools)

    # Turn 2
    client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Let me search for that."},
        {"role": "tool", "content": "Python is a high-level language...", "tool_call_id": "1"},
    ], tools=tools)

    trace = tracer.to_trace()
    types = [s.type for s in trace.steps]
    assert types == ["thought", "tool_call", "observation", "result"]
    assert tracer._available_tools == ["web_search"]
    assert tracer._model == "gpt-4o"
