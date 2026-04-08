"""CLI entry point for AgentDiag."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from agentdiag import __version__
from agentdiag.adapters import detect_and_parse
from agentdiag.detectors import run_all
from agentdiag.report import build_report, render_json, render_rich

app = typer.Typer(
    name="agentdiag",
    help="Diagnose why AI agents fail. Lightweight CLI for agent trace analysis.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"agentdiag {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """AgentDiag — Diagnose why AI agents fail."""


@app.command()
def analyze(
    files: Annotated[
        list[Path],
        typer.Argument(help="Trace JSON file(s) to analyze."),
    ],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: rich (default) or json."),
    ] = "rich",
) -> None:
    """Analyze agent trace(s) and report findings."""
    for file_path in files:
        if not file_path.exists():
            console.print(f"[red]Error:[/red] File not found: {file_path}")
            raise typer.Exit(1)

        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON in {file_path}: {e}")
            raise typer.Exit(1)

        try:
            trace = detect_and_parse(raw)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        findings = run_all(trace)
        report = build_report(trace, findings)

        if format == "json":
            console.print(render_json(report), highlight=False)
        else:
            render_rich(report, console)
