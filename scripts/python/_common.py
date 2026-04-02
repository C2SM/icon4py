"""Shared helpers for Python dev-scripts."""

from __future__ import annotations

import subprocess
import pathlib
from typing import Final

import rich
import typer


PY_SCRIPTS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().absolute().parent
SCRIPTS_DIR: Final[pathlib.Path] = PY_SCRIPTS_DIR.parent
REPO_ROOT: Final[pathlib.Path] = SCRIPTS_DIR.parent


def run_or_fail(
    cmd: list[str],
    **kwargs,
) -> subprocess.CompletedProcess[bytes]:
    """Run *cmd* and exit with its return code on failure.

    All *kwargs* are forwarded to :func:`subprocess.run`.
    """
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        rich.print(
            f"[bold red]Error[/bold red] Command failed (rc={result.returncode}): {' '.join(cmd)}",
        )
        raise typer.Exit(result.returncode)
    return result
