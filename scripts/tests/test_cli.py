"""Tests for CLI structure and sub-command registration."""

from __future__ import annotations

from typer.testing import CliRunner

from py._cli import app

runner = CliRunner()


def test_help_displays_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Python sub-commands
    assert "lint" in result.output
    assert "migrate-db" in result.output
    # Shell sub-commands (auto-discovered)
    assert "setup-env" in result.output
    assert "cleanup" in result.output
    # Just namespace
    assert "just" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output
