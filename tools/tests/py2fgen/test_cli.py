# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from icon4pytools.py2fgen.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def wrapper_module():
    return "icon4pytools.py2fgen.wrappers.simple"


@pytest.fixture
def diffusion_module():
    return "icon4pytools.py2fgen.wrappers.diffusion_test_case"


def run_test_case(
    cli,
    module: str,
    function: str,
    plugin_name: str,
    backend: str,
    samples_path: Path,
    fortran_driver: str,
    extra_compiler_flags: tuple[str, ...] = (),
):
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module, function, plugin_name, "--gt4py-backend", backend, "-d"])
        assert result.exit_code == 0, "CLI execution failed"

        try:
            compile_fortran_code(plugin_name, samples_path, fortran_driver, extra_compiler_flags)
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Compilation failed: {e}")

        try:
            fortran_result = run_fortran_executable(plugin_name)
            assert "passed" in fortran_result.stdout
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")


def compile_fortran_code(
    plugin_name: str, samples_path: Path, fortran_driver: str, extra_compiler_flags: tuple[str, ...]
):
    subprocess.run(["gfortran", "-c", f"{plugin_name}.f90", "."], check=True)
    subprocess.run(
        [
            "gfortran",
            "-cpp",
            "-I.",
            "-Wl,-rpath=.",
            "-L.",
            f"{plugin_name}.f90",
            str(samples_path / f"{fortran_driver}.f90"),
            f"-l{plugin_name}",
            "-o",
            plugin_name,
        ]
        + [f for f in extra_compiler_flags],
        check=True,
    )


def run_fortran_executable(plugin_name: str):
    return subprocess.run([f"./{plugin_name}"], capture_output=True, text=True, check=True)


@pytest.mark.parametrize(
    "backend, extra_flags",
    [
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
        ("ROUNDTRIP", ""),
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
        ("ROUNDTRIP", ""),
    ],
)
def test_py2fgen_compilation_and_execution_square(
    cli_runner, backend, samples_path, wrapper_module, extra_flags
):
    run_test_case(
        cli_runner,
        wrapper_module,
        "square,square_from_function",
        "square_plugin",
        backend,
        samples_path,
        "test_square",
        extra_flags,
    )


@pytest.mark.parametrize("backend", ("CPU", "ROUNDTRIP"))
def test_py2fgen_compilation_and_execution_multi_return(
    cli_runner, backend, samples_path, wrapper_module
):
    run_test_case(
        cli_runner,
        wrapper_module,
        "multi_return",
        "multi_return_plugin",
        backend,
        samples_path,
        "test_multi_return",
    )
