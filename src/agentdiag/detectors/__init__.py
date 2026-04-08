"""Detector registry — discover and run all detectors."""

from __future__ import annotations

from agentdiag.detectors.base import BaseDetector
from agentdiag.detectors.hallucination import HallucinatedSuccessDetector
from agentdiag.detectors.loop import LoopDetector
from agentdiag.detectors.premature_stop import PrematureStopDetector
from agentdiag.detectors.recovery import RecoveryFailureDetector
from agentdiag.detectors.tool_misuse import ToolMisuseDetector
from agentdiag.schema import Finding, Trace

ALL_DETECTORS: list[type[BaseDetector]] = [
    LoopDetector,
    ToolMisuseDetector,
    RecoveryFailureDetector,
    PrematureStopDetector,
    HallucinatedSuccessDetector,
]


def run_all(trace: Trace) -> list[Finding]:
    """Run every registered detector against a trace and return all findings."""
    findings: list[Finding] = []
    for detector_cls in ALL_DETECTORS:
        detector = detector_cls()
        findings.extend(detector.detect(trace))
    # Sort by step range start
    findings.sort(key=lambda f: f.step_range[0])
    return findings
