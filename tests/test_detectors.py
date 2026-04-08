"""Tests for all detectors."""

import json
from pathlib import Path

from agentdiag.adapters import detect_and_parse
from agentdiag.detectors import run_all
from agentdiag.detectors.hallucination import HallucinatedSuccessDetector
from agentdiag.detectors.loop import LoopDetector
from agentdiag.detectors.premature_stop import PrematureStopDetector
from agentdiag.detectors.recovery import RecoveryFailureDetector
from agentdiag.detectors.tool_misuse import ToolMisuseDetector

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    raw = json.loads((FIXTURES / name).read_text())
    return detect_and_parse(raw)


# --- Loop Detector ---


def test_loop_detected():
    trace = _load("loop_trace.json")
    findings = LoopDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].detector == "LOOP"
    assert findings[0].severity == "high"


def test_no_loop_in_clean():
    trace = _load("clean_trace.json")
    findings = LoopDetector().detect(trace)
    assert len(findings) == 0


# --- Tool Misuse Detector ---


def test_tool_misuse_detected():
    trace = _load("tool_misuse_trace.json")
    findings = ToolMisuseDetector().detect(trace)
    assert len(findings) == 2
    assert all(f.detector == "TOOL_MISUSE" for f in findings)


def test_no_misuse_in_clean():
    trace = _load("clean_trace.json")
    findings = ToolMisuseDetector().detect(trace)
    assert len(findings) == 0


# --- Recovery Failure Detector ---


def test_recovery_failure_detected():
    trace = _load("recovery_failure_trace.json")
    findings = RecoveryFailureDetector().detect(trace)
    assert len(findings) >= 1
    assert all(f.detector == "RECOVERY_FAILURE" for f in findings)


def test_no_recovery_issue_in_clean():
    trace = _load("clean_trace.json")
    findings = RecoveryFailureDetector().detect(trace)
    assert len(findings) == 0


# --- Premature Stop Detector ---


def test_premature_stop_detected():
    trace = _load("loop_trace.json")
    findings = PrematureStopDetector().detect(trace)
    assert len(findings) == 1
    assert findings[0].detector == "PREMATURE_STOP"


def test_no_premature_stop_in_clean():
    trace = _load("clean_trace.json")
    findings = PrematureStopDetector().detect(trace)
    assert len(findings) == 0


# --- Hallucinated Success Detector ---


def test_hallucinated_success_not_triggered_on_honest_failure():
    trace = _load("loop_trace.json")
    findings = HallucinatedSuccessDetector().detect(trace)
    # The loop trace says "I was unable..." — not a hallucinated success
    assert len(findings) == 0


# --- run_all integration ---


def test_run_all_loop_trace():
    trace = _load("loop_trace.json")
    findings = run_all(trace)
    detectors_found = {f.detector for f in findings}
    assert "LOOP" in detectors_found
    assert "RECOVERY_FAILURE" in detectors_found


def test_run_all_clean_trace():
    trace = _load("clean_trace.json")
    findings = run_all(trace)
    assert len(findings) == 0
