"""Tests for trace adapters and auto-detection."""

import json
from pathlib import Path

from agentdiag.adapters import detect_and_parse
from agentdiag.adapters.langchain import LangChainAdapter
from agentdiag.adapters.openai_sdk import OpenAIAgentsAdapter
from agentdiag.adapters.raw import RawAdapter

FIXTURES = Path(__file__).parent / "fixtures"


def _load_raw(name: str):
    return json.loads((FIXTURES / name).read_text())


# --- Raw Adapter ---


def test_raw_can_handle():
    raw = _load_raw("clean_trace.json")
    assert RawAdapter.can_handle(raw) is True


def test_raw_parse():
    raw = _load_raw("clean_trace.json")
    trace = RawAdapter().parse(raw)
    assert trace.task == "What is the weather in San Francisco today?"
    assert len(trace.steps) == 4
    assert trace.outcome == "success"


# --- LangChain Adapter ---


def test_langchain_can_handle():
    raw = _load_raw("langchain_trace.json")
    assert LangChainAdapter.can_handle(raw) is True


def test_langchain_does_not_handle_raw():
    raw = _load_raw("clean_trace.json")
    assert LangChainAdapter.can_handle(raw) is False


def test_langchain_parse():
    raw = _load_raw("langchain_trace.json")
    trace = LangChainAdapter().parse(raw)
    assert trace.task == "What is the population of Tokyo?"
    assert trace.framework == "langchain"
    assert trace.outcome == "failure"
    assert len(trace.steps) > 0
    tool_calls = [s for s in trace.steps if s.type == "tool_call"]
    assert len(tool_calls) == 3
    assert all(s.tool_name == "web_search" for s in tool_calls)


# --- OpenAI Agents SDK Adapter ---


def test_openai_can_handle():
    raw = _load_raw("openai_sdk_trace.json")
    assert OpenAIAgentsAdapter.can_handle(raw) is True


def test_openai_does_not_handle_raw():
    raw = _load_raw("clean_trace.json")
    assert OpenAIAgentsAdapter.can_handle(raw) is False


def test_openai_parse():
    raw = _load_raw("openai_sdk_trace.json")
    trace = OpenAIAgentsAdapter().parse(raw)
    assert trace.task == "Send a Slack message to #general saying hello"
    assert trace.framework == "openai_agents_sdk"
    assert trace.outcome == "failure"
    tool_calls = [s for s in trace.steps if s.type == "tool_call"]
    assert len(tool_calls) == 2


# --- Auto-detection ---


def test_autodetect_raw():
    raw = _load_raw("clean_trace.json")
    trace = detect_and_parse(raw)
    assert trace.task == "What is the weather in San Francisco today?"


def test_autodetect_langchain():
    raw = _load_raw("langchain_trace.json")
    trace = detect_and_parse(raw)
    assert trace.framework == "langchain"


def test_autodetect_openai():
    raw = _load_raw("openai_sdk_trace.json")
    trace = detect_and_parse(raw)
    assert trace.framework == "openai_agents_sdk"
