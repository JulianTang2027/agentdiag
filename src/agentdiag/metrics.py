"""Compute aggregate metrics from a trace and its findings."""

from __future__ import annotations

from typing import Any

from agentdiag.schema import Finding, Trace


def compute_metrics(trace: Trace, findings: list[Finding]) -> dict[str, Any]:
    """Compute diagnostic metrics for a trace."""
    tool_calls = [s for s in trace.steps if s.type == "tool_call"]
    errors = [s for s in trace.steps if s.type == "observation" and s.error]

    # --- Recovery Rate ---
    # For each error, check if the next tool call changed strategy
    recoveries = 0
    recovery_opportunities = 0
    for i, step in enumerate(trace.steps):
        if not (step.type == "observation" and step.error):
            continue
        # Find preceding tool call
        prev_call = None
        for j in range(i - 1, -1, -1):
            if trace.steps[j].type == "tool_call":
                prev_call = trace.steps[j]
                break
        # Find next tool call
        next_call = None
        for k in range(i + 1, len(trace.steps)):
            if trace.steps[k].type == "tool_call":
                next_call = trace.steps[k]
                break
        if prev_call and next_call:
            recovery_opportunities += 1
            changed = (
                prev_call.tool_name != next_call.tool_name
                or prev_call.tool_args != next_call.tool_args
            )
            if changed:
                recoveries += 1

    recovery_rate = (
        recoveries / recovery_opportunities if recovery_opportunities > 0 else None
    )

    # --- Tool Accuracy ---
    # Fraction of tool calls that used a valid tool name
    if trace.available_tools is not None and tool_calls:
        available = set(trace.available_tools)
        valid_calls = sum(1 for s in tool_calls if s.tool_name in available)
        tool_accuracy = valid_calls / len(tool_calls)
    else:
        tool_accuracy = None

    # --- Loop Count ---
    loop_count = sum(1 for f in findings if f.detector == "LOOP")

    # --- Failure Density ---
    failure_density = len(findings) / len(trace.steps) if trace.steps else 0.0

    return {
        "total_steps": len(trace.steps),
        "tool_calls": len(tool_calls),
        "errors": len(errors),
        "recovery_rate": recovery_rate,
        "tool_accuracy": tool_accuracy,
        "loop_count": loop_count,
        "finding_count": len(findings),
        "failure_density": round(failure_density, 3),
    }
