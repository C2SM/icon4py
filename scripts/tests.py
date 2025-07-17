# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Manage test infrastructure."""

import enum
import pathlib
import subprocess

import rich
import typer

from . import _common as common


class ExitCode(enum.IntEnum):
    """Exit codes for the script."""

    INVALID_COMMAND_OPTIONS = 1
    GIT_LS_ERROR = 2
    MISSING_INIT_FILES = 3

cli = typer.Typer(help=__doc__)


@cli.command(name="check-tests")
def check_tests(
    fix: bool = typer.Option(False, "--fix", "-f", help="Automatically create missing __init__.py files.")
):
    """Check if all 'tests' subpackages have '__init__.py' files."""
    root_dirs: list[pathlib.Path] = [common.REPO_ROOT / "model", common.REPO_ROOT / "tools"]
    missing = 0
    init_py_content = ""

    if fix:
        init_py_content = "\n".join(
            f"# {line}" if line else "#"
            for line in (common.REPO_ROOT / "HEADER.txt").read_text().splitlines()
        ) + "\n\n"

    for root_dir in root_dirs:
        prefix_len = len(root_dir.parts)
        for (dir_path, _, file_names) in root_dir.walk(top_down=True):
            if dir_path.name.startswith((".", "__")):
                # Skip hidden and special directories (e.g. '__pycache__')
                continue
            local_parts = dir_path.parts[prefix_len:]
            within_tests_package = "tests" in local_parts[:-1]
            if within_tests_package:
                rich.print(f"Checking '{'/'.join(local_parts)}'")

            if within_tests_package and '__init__.py' not in file_names:
                missing += 1
                rich.print("  [red]-> Missing '__init__.py' file[/red]")
                if fix:
                    rich.print(f"  [yellow]-> Creating '__init__.py' in {dir_path}[/yellow]")
                    (dir_path /'__init__.py').write_text(init_py_content)

    rich.print()
    if missing:
        if fix:
            rich.print(f"[yellow]FIXED:[/yellow] {missing} missing '__init__.py' files have been created.")
        else:
            rich.print(f"[red]ERROR:[/red] {missing} '__init__.py' files are missing!")
            raise typer.Exit(code=ExitCode.MISSING_INIT_FILES)

    else:
        rich.print(f"[green]OK:[/green] Tests structure seems correct.")



if __name__ == "__main__":
    cli()