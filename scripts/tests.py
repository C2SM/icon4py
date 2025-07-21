# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Manage test infrastructure."""

import ast
import enum
import pathlib

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
    fix: bool = typer.Option(
        False, "--fix", "-f", help="Automatically create missing __init__.py files."
    ),
):
    """Check if all 'tests' subpackages have proper '__init__.py' files."""
    root_dirs: list[pathlib.Path] = [common.REPO_ROOT / "model", common.REPO_ROOT / "tools"]
    violations = 0
    ns_init_py_content = init_py_content = ""

    if fix:
        init_py_content = (
            "\n".join(
                f"# {line}" if line else "#"
                for line in [*(common.REPO_ROOT / "HEADER.txt").read_text().splitlines(), "\n"]
            )
            + "\n"
        )
        ns_init_py_content = (
            init_py_content + "# legacy namespace package for tests\n"
            '__path__ = __import__("pkgutil").extend_path(__path__, __name__)\n'
        )

    for root_dir in root_dirs:
        prefix_len = len(root_dir.parts)
        for dir_path, _, file_names in root_dir.walk(top_down=True):
            if dir_path.name.startswith((".", "__")):
                # Skip hidden and special directories (e.g. '__pycache__')
                continue

            local_parts = dir_path.parts[prefix_len:]
            if "tests" in local_parts:
                if is_ns_pkg := (local_parts[-1] == "tests"):
                    rich.print(f"Checking '{'/'.join(local_parts)}' (namespace package)")
                    fix_content = ns_init_py_content
                else:
                    rich.print(f"Checking '{'/'.join(local_parts)}'")
                    fix_content = init_py_content

                if "__init__.py" not in file_names:
                    violations += 1
                    rich.print("  [red]-> Missing '__init__.py' file[/red]")
                    if fix:
                        rich.print(f"  [yellow]-> Creating '__init__.py' in {dir_path}[/yellow]")
                        (dir_path / "__init__.py").write_text(fix_content)
                elif is_ns_pkg:
                    file_ast = ast.parse((dir_path / "__init__.py").read_text())
                    template_ast = ast.parse('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
                    if ast.dump(file_ast, annotate_fields=False, include_attributes=False) != ast.dump(template_ast, annotate_fields=False, include_attributes=False):
                        violations += 1
                        rich.print("  [red]-> '__init__.py' does not match namespace package template[/red]")
                        if fix:
                            rich.print(f"  [yellow]-> Fixing '__init__.py' in {dir_path}[/yellow]")
                            (dir_path / "__init__.py").write_text(fix_content)

    rich.print()
    if violations:
        if fix:
            rich.print(
                f"[yellow]FIXED:[/yellow] {violations} missing '__init__.py' files have been created."
            )
        else:
            rich.print(f"[red]ERROR:[/red] {violations} '__init__.py' files are missing!")
            raise typer.Exit(code=ExitCode.MISSING_INIT_FILES)

    else:
        rich.print(f"[green]OK:[/green] Tests structure seems correct.")


if __name__ == "__main__":
    cli()
