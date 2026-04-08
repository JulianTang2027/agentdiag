"""Tests for the trace schema."""

from agentdiag.schema import DiagnosticReport, Finding, Step, Trace


def test_step_minimal():
    step = Step(index=0, type="thought", content="thinking")
    assert step.index == 0
    assert step.tool_name is None
    assert step.error is False


def test_step_tool_call():
    step = Step(
        index=1,
        type="tool_call",
        content="calling",
        tool_name="search",
        tool_args={"q": "test"},
    )
    assert step.tool_name == "search"
    assert step.tool_args == {"q": "test"}


def test_trace_minimal():
    trace = Trace(
        task="do something",
        steps=[Step(index=0, type="thought", content="thinking")],
    )
    assert trace.outcome == "unknown"
    assert trace.model is None
    assert trace.metadata == {}


def test_finding():
    f = Finding(
        detector="LOOP",
        severity="high",
        step_range=(1, 5),
        summary="looped",
        suggestion="stop looping",
    )
    assert f.step_range == (1, 5)


def test_diagnostic_report():
    trace = Trace(task="test", steps=[])
    report = DiagnosticReport(trace=trace)
    assert report.findings == []
    assert report.metrics == {}
