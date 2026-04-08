"""Report rendering — rich terminal output and JSON export."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentdiag.metrics import compute_metrics
from agentdiag.schema import DiagnosticReport, Finding, Trace

SEVERITY_COLORS = {"high": "red", "medium": "yellow", "low": "blue"}


def build_report(trace: Trace, findings: list[Finding]) -> DiagnosticReport:
    """Build a full diagnostic report with metrics."""
    metrics = compute_metrics(trace, findings)
    return DiagnosticReport(trace=trace, findings=findings, metrics=metrics)


def render_rich(report: DiagnosticReport, console: Console) -> None:
    """Render a diagnostic report as beautiful rich terminal output."""
    trace = report.trace
    findings = report.findings
    metrics = report.metrics

    # --- Header ---
    outcome_color = "green" if trace.outcome == "success" else "red"
    header = Text()
    header.append("Task:    ", style="bold")
    header.append(f"{trace.task}\n")
    header.append("Model:   ", style="bold")
    header.append(f"{trace.model or 'unknown'}\n")
    header.append("Steps:   ", style="bold")
    header.append(f"{metrics['total_steps']}\n")
    header.append("Outcome: ", style="bold")
    header.append(f"{trace.outcome.upper()}", style=f"bold {outcome_color}")

    console.print(Panel(header, title="[bold cyan]AgentDiag Report[/bold cyan]", border_style="cyan"))

    if not findings:
        console.print(Panel("[bold green]No issues detected.[/bold green]", border_style="green"))
        return

    # --- Findings Table ---
    table = Table(title="Findings", border_style="yellow", title_style="bold yellow")
    table.add_column("Severity", style="bold", width=8)
    table.add_column("Detector", style="bold", width=20)
    table.add_column("Steps", width=8)
    table.add_column("Summary")

    for f in findings:
        sev_color = SEVERITY_COLORS[f.severity]
        table.add_row(
            f"[{sev_color}]{f.severity.upper()}[/{sev_color}]",
            f.detector,
            f"{f.step_range[0]}-{f.step_range[1]}",
            f.summary,
        )

    console.print(table)

    # --- Metrics Bar ---
    metrics_parts: list[str] = []

    if metrics["recovery_rate"] is not None:
        rate = metrics["recovery_rate"]
        color = "green" if rate > 0.5 else "yellow" if rate > 0 else "red"
        metrics_parts.append(f"Recovery Rate: [{color}]{rate:.0%}[/{color}]")

    if metrics["tool_accuracy"] is not None:
        acc = metrics["tool_accuracy"]
        color = "green" if acc > 0.8 else "yellow" if acc > 0.5 else "red"
        metrics_parts.append(f"Tool Accuracy: [{color}]{acc:.0%}[/{color}]")

    if metrics["loop_count"] > 0:
        metrics_parts.append(f"Loops: [red]{metrics['loop_count']}[/red]")

    metrics_parts.append(f"Findings: {metrics['finding_count']}")

    if metrics_parts:
        console.print(Panel("  |  ".join(metrics_parts), title="[bold]Metrics[/bold]", border_style="dim"))

    # --- Suggestions ---
    console.print("\n[bold]Suggestions:[/bold]")
    for f in findings:
        console.print(f"  [dim]Steps {f.step_range[0]}-{f.step_range[1]}:[/dim] {f.suggestion}")
    console.print()


def render_json(report: DiagnosticReport) -> str:
    """Render a diagnostic report as JSON for CI/CD piping."""
    data: dict[str, Any] = {
        "task": report.trace.task,
        "model": report.trace.model,
        "framework": report.trace.framework,
        "outcome": report.trace.outcome,
        "total_steps": len(report.trace.steps),
        "metrics": report.metrics,
        "findings": [
            {
                "detector": f.detector,
                "severity": f.severity,
                "step_range": list(f.step_range),
                "summary": f.summary,
                "suggestion": f.suggestion,
            }
            for f in report.findings
        ],
    }
    return json.dumps(data, indent=2)
