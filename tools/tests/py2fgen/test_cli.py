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


def run_test_case(
    cli,
    module: str,
    function: str,
    backend: str,
    samples_path: Path,
    fortran_driver: str,
    extra_compiler_flags: tuple[str, ...] = (),
):
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module, function, "--gt4py-backend", backend, "-d"])
        assert result.exit_code == 0, "CLI execution failed"

        try:
            compile_fortran_code(function, samples_path, fortran_driver, extra_compiler_flags)
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Compilation failed: {e}")

        try:
            fortran_result = run_fortran_executable(function)
            assert "passed" in fortran_result.stdout
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")


def compile_fortran_code(
    function: str, samples_path: Path, fortran_driver: str, extra_compiler_flags: tuple[str, ...]
):
    subprocess.run(["gfortran", "-c", f"{function}_plugin.f90", "."], check=True)
    subprocess.run(
        [
            "gfortran",
            "-cpp",
            "-I.",
            "-Wl,-rpath=.",
            "-L.",
            f"{function}_plugin.f90",
            str(samples_path / f"{fortran_driver}.f90"),
            f"-l{function}_plugin",
            "-o",
            function,
        ]
        + [f for f in extra_compiler_flags],
        check=True,
    )


def run_fortran_executable(function: str):
    return subprocess.run([f"./{function}"], capture_output=True, text=True, check=True)


@pytest.mark.parametrize("backend", ("CPU", "ROUNDTRIP"))
def test_py2fgen_compilation_and_execution_square(
    cli_runner, backend, samples_path, wrapper_module
):
    run_test_case(
        cli_runner,
        wrapper_module,
        "square",
        backend,
        samples_path,
        "test_square",
    )


@pytest.mark.parametrize("backend", ("CPU", "ROUNDTRIP"))
def test_py2fgen_compilation_and_execution_square_from_function(
    cli_runner, backend, samples_path, wrapper_module
):
    run_test_case(
        cli_runner,
        wrapper_module,
        "square_from_function",
        backend,
        samples_path,
        "test_square",
        ("-DUSE_SQUARE_FROM_FUNCTION",),
    )


@pytest.mark.parametrize("backend", ("CPU", "ROUNDTRIP"))
def test_py2fgen_compilation_and_execution_multi_return(
    cli_runner, backend, samples_path, wrapper_module
):
    run_test_case(
        cli_runner,
        wrapper_module,
        "multi_return",
        backend,
        samples_path,
        "test_multi_return",
    )
