"""Tests for the Tracer SDK."""

import json
from pathlib import Path

from agentdiag import Tracer


def test_basic_step_logging():
    t = Tracer(task="test task", model="gpt-4o")
    t.thought("thinking")
    t.tool_call("search", args={"q": "hello"})
    t.observation("result here")
    t.result("done")

    trace = t.to_trace()
    assert trace.task == "test task"
    assert trace.model == "gpt-4o"
    assert len(trace.steps) == 4
    assert trace.steps[0].type == "thought"
    assert trace.steps[1].type == "tool_call"
    assert trace.steps[1].tool_name == "search"
    assert trace.steps[1].tool_args == {"q": "hello"}
    assert trace.steps[2].type == "observation"
    assert trace.steps[3].type == "result"


def test_auto_incrementing_index():
    t = Tracer(task="test")
    t.thought("a")
    t.thought("b")
    t.thought("c")
    trace = t.to_trace()
    assert [s.index for s in trace.steps] == [0, 1, 2]


def test_timestamps_are_set():
    t = Tracer(task="test")
    t.thought("a")
    assert t.to_trace().steps[0].timestamp is not None


def test_set_outcome():
    t = Tracer(task="test")
    t.result("done")
    assert t.to_trace().outcome == "unknown"

    t.set_outcome("failure")
    assert t.to_trace().outcome == "failure"

    t.set_outcome("success")
    assert t.to_trace().outcome == "success"


def test_available_tools_passed_through():
    t = Tracer(task="test", available_tools=["search", "write"])
    trace = t.to_trace()
    assert trace.available_tools == ["search", "write"]


def test_diagnose_clean_trace():
    t = Tracer(task="get weather", model="gpt-4o")
    t.thought("checking weather")
    t.tool_call("get_weather", args={"city": "SF"})
    t.observation("72F sunny")
    t.result("It's 72F and sunny in SF")
    t.set_outcome("success")

    report = t.diagnose(format="quiet")
    assert len(report.findings) == 0


def test_diagnose_catches_loop():
    t = Tracer(task="failing task", model="gpt-4o")
    t.set_outcome("failure")

    for _ in range(4):
        t.tool_call("bad_api", args={"x": 1})
        t.observation("Error: timeout", error=True)

    t.result("I failed")
    report = t.diagnose(format="quiet")

    detectors = {f.detector for f in report.findings}
    assert "LOOP" in detectors
    assert "RECOVERY_FAILURE" in detectors


def test_diagnose_catches_tool_misuse():
    t = Tracer(
        task="send email",
        available_tools=["send_email", "read_inbox"],
    )
    t.tool_call("gmail_send", args={"to": "a@b.com"})
    t.observation("Error: tool not found", error=True)
    t.result("could not send")
    t.set_outcome("failure")

    report = t.diagnose(format="quiet")
    detectors = {f.detector for f in report.findings}
    assert "TOOL_MISUSE" in detectors


def test_save_and_reload(tmp_path):
    t = Tracer(task="save test", model="claude-3")
    t.thought("thinking")
    t.tool_call("search", args={"q": "test"})
    t.observation("found it")
    t.result("done")
    t.set_outcome("success")

    out = tmp_path / "trace.json"
    t.save(out)

    data = json.loads(out.read_text())
    assert data["task"] == "save test"
    assert data["model"] == "claude-3"
    assert len(data["steps"]) == 4
    assert data["outcome"] == "success"


def test_context_manager(capsys):
    with Tracer(task="ctx test", model="gpt-4o") as t:
        t.tool_call("search", args={"q": "test"})
        t.observation("ok")
        t.result("done")
        t.set_outcome("success")

    # Context manager should have printed output (rich report)
    captured = capsys.readouterr()
    assert "AgentDiag Report" in captured.out


def test_tool_call_default_content():
    t = Tracer(task="test")
    t.tool_call("my_tool")
    trace = t.to_trace()
    assert trace.steps[0].content == "Tool call: my_tool"


def test_tool_call_custom_content():
    t = Tracer(task="test")
    t.tool_call("my_tool", content="doing something special")
    trace = t.to_trace()
    assert trace.steps[0].content == "doing something special"


def test_metadata_passed_through():
    t = Tracer(task="test", metadata={"env": "staging", "run_id": "abc123"})
    trace = t.to_trace()
    assert trace.metadata == {"env": "staging", "run_id": "abc123"}
