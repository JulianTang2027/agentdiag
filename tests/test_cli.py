"""Tests for the CLI."""

from pathlib import Path

from typer.testing import CliRunner

from agentdiag.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_analyze_clean():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "clean_trace.json")])
    assert result.exit_code == 0
    assert "No issues detected" in result.output


def test_analyze_loop():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "loop_trace.json")])
    assert result.exit_code == 0
    assert "LOOP" in result.output
    assert "RECOVERY_FAILURE" in result.output


def test_analyze_json_format():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "tool_misuse_trace.json"), "--format", "json"])
    assert result.exit_code == 0
    assert '"detector": "TOOL_MISUSE"' in result.output


def test_analyze_missing_file():
    result = runner.invoke(app, ["analyze", "nonexistent.json"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_analyze_multiple_files():
    result = runner.invoke(app, [
        "analyze",
        str(FIXTURES / "clean_trace.json"),
        str(FIXTURES / "loop_trace.json"),
    ])
    assert result.exit_code == 0
    assert "No issues detected" in result.output
    assert "LOOP" in result.output


def test_analyze_langchain():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "langchain_trace.json")])
    assert result.exit_code == 0
    assert "LOOP" in result.output


def test_analyze_openai_sdk():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "openai_sdk_trace.json")])
    assert result.exit_code == 0
    assert "TOOL_MISUSE" in result.output
