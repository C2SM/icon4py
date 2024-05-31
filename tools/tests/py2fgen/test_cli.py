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
import os
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
    compiler: str = "gfortran",
    extra_compiler_flags: tuple[str, ...] = (),
    expected_error_code: int = 0,
):
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module, function, plugin_name, "-b", backend, "--limited-area"])
        assert result.exit_code == 0, "CLI execution failed"

        try:
            compile_fortran_code(
                plugin_name, samples_path, fortran_driver, compiler, extra_compiler_flags
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Compilation failed: {e}\n{e.stderr}\n{e.stdout}")

        try:
            fortran_result = run_fortran_executable(plugin_name)
            if expected_error_code == 0:
                assert "passed" in fortran_result.stdout
            else:
                assert "failed" in fortran_result.stdout
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")


def compile_fortran_code(
    plugin_name: str,
    samples_path: Path,
    fortran_driver: str,
    compiler: str,
    extra_compiler_flags: tuple[str, ...],
):
    command = [
        f"{compiler}",
        "-cpp",
        "-I.",
        "-Wl,-rpath=.",
        "-L.",
        f"{plugin_name}.f90",
        str(samples_path / f"{fortran_driver}.f90"),
        f"-l{plugin_name}",
        "-o",
        plugin_name,
    ] + [f for f in extra_compiler_flags]
    subprocess.run(command, check=True, capture_output=True, text=True)


def run_fortran_executable(plugin_name: str):
    try:
        result = subprocess.run([f"./{plugin_name}"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # If an error occurs, use the exception's `stdout` and `stderr`.
        result = e
    return result


@pytest.mark.parametrize(
    "backend, extra_flags",
    [
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
        ("ROUNDTRIP", ""),
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
        ("ROUNDTRIP", ""),
    ],
)
def test_py2fgen_compilation_and_execution_square_cpu(
    cli_runner, backend, samples_path, wrapper_module, extra_flags
):
    """Tests embedding Python functions, and GT4Py program directly.
    Also tests embedding multiple functions in one shared library.
    """
    run_test_case(
        cli_runner,
        wrapper_module,
        "square,square_from_function",
        "square_plugin",
        backend,
        samples_path,
        "test_square",
        extra_compiler_flags=extra_flags,
    )


def test_py2fgen_python_error_propagation_to_fortran(cli_runner, samples_path, wrapper_module):
    """Tests that Exceptions triggered in Python propagate an error code (1) up to Fortran."""
    run_test_case(
        cli_runner,
        wrapper_module,
        "square_error",
        "square_plugin",
        "ROUNDTRIP",
        samples_path,
        "test_square",
        extra_compiler_flags=("-DUSE_SQUARE_ERROR",),
        expected_error_code=1,
    )


@pytest.mark.skipif(os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI.")
@pytest.mark.parametrize(
    "function_name, plugin_name, test_name, backend, extra_flags",
    [
        ("square", "square_plugin", "test_square", "GPU", ("-acc", "-Minfo=acc")),
    ],
)
def test_py2fgen_compilation_and_execution_gpu(
    cli_runner,
    function_name,
    plugin_name,
    test_name,
    backend,
    samples_path,
    wrapper_module,
    extra_flags,
):
    run_test_case(
        cli_runner,
        wrapper_module,
        function_name,
        plugin_name,
        backend,
        samples_path,
        test_name,
        "nvfortran",
        extra_flags,
    )


@pytest.mark.parametrize(
    "backend, extra_flags",
    [
        ("CPU", ("-DPROFILE_SQUARE_FROM_FUNCTION",)),
    ],
)
def test_py2fgen_compilation_and_profiling(
    cli_runner, backend, samples_path, wrapper_module, extra_flags
):
    """Test profiling using cProfile of the generated wrapper."""
    run_test_case(
        cli_runner,
        wrapper_module,
        "square_from_function,profile_enable,profile_disable",
        "square_plugin",
        backend,
        samples_path,
        "test_square",
        extra_compiler_flags=extra_flags,
    )


@pytest.mark.skipif(os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI.")
def test_py2fgen_compilation_and_execution_diffusion_gpu(
    cli_runner,
    samples_path,
):
    run_test_case(
        cli_runner,
        "icon4pytools.py2fgen.wrappers.diffusion",
        "diffusion_init,diffusion_run,profile_enable,profile_disable",
        "diffusion_plugin",
        "GPU",
        samples_path,
        "test_diffusion",
        "nvfortran",
        ("-acc", "-Minfo=acc"),
    )


def test_py2fgen_compilation_and_execution_diffusion(cli_runner, samples_path):
    run_test_case(
        cli_runner,
        "icon4pytools.py2fgen.wrappers.diffusion",
        "diffusion_init,diffusion_run,profile_enable,profile_disable",
        "diffusion_plugin",
        "CPU",
        samples_path,
        "test_diffusion",
    )
