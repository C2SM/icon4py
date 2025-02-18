# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import difflib
import pathlib

import pytest
from click.testing import CliRunner

from icon4py.tools.py2fgen import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


def reference_path(bindings_name: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve() / "references" / bindings_name


def actual_path(bindings_name: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve() / "references_new" / bindings_name


def invoke_cli(cli_runner, module, function, plugin_name, backend, path):
    cli_args = [module, function, plugin_name, "-b", backend, "-d", "-o", path]
    result = cli_runner.invoke(cli.main, cli_args)
    assert result.exit_code == 0, "CLI execution failed"


def diff(reference: pathlib.Path, actual: pathlib.Path):
    with open(reference, "r") as f:
        reference_lines = f.readlines()
    with open(actual, "r") as f:
        actual_lines = f.readlines()
    result = difflib.context_diff(reference_lines, actual_lines)

    clean = True
    for line in result:
        print(line)
        clean = False

    return clean


def check_generated_files(bindings_name: str) -> None:
    for suffix in [".h", ".f90", ".py"]:
        assert diff(
            reference_path(bindings_name) / f"{bindings_name}{suffix}",
            actual_path(bindings_name) / f"{bindings_name}{suffix}",
        )


@pytest.mark.parametrize(
    "bindings_name, module, functions",
    [
        (
            "diffusion",
            "icon4py.tools.py2fgen.wrappers.diffusion_wrapper",
            "diffusion_run, diffusion_init, grid_init_diffusion",
        ),
        (
            "dycore",
            "icon4py.tools.py2fgen.wrappers.dycore_wrapper",
            "solve_nh_run, solve_nh_init, grid_init",
        ),
    ],
    ids=["diffusion", "dycore"],
)
def test_references(cli_runner, bindings_name, module, functions):
    invoke_cli(
        cli_runner,
        module,
        functions,
        bindings_name,
        "GPU",
        actual_path(bindings_name),
    )
    check_generated_files(bindings_name)
