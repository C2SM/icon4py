# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

import pytest
from click.testing import CliRunner

from icon4pytools.py2fgen.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def square_wrapper_module():
    return "icon4pytools.py2fgen.wrappers.simple"


def compile_fortran_code(
    plugin_name, samples_path, fortran_driver, compiler, extra_compiler_flags, backend
):
    shared_library = f"{plugin_name}_{backend.lower()}"
    command = [
        f"{compiler}",
        "-cpp",
        "-I.",
        "-Wl,-rpath=.",
        "-L.",
        f"{plugin_name}.f90",
        str(samples_path / f"{fortran_driver}.f90"),
        f"-l{shared_library}",
        "-o",
        plugin_name,
        *list(extra_compiler_flags),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def run_fortran_executable(plugin_name, env):
    try:
        result = subprocess.run(
            [f"./{plugin_name}"], capture_output=True, text=True, check=True, env=env
        )
    except subprocess.CalledProcessError as e:
        # If an error occurs, use the exception's `stdout` and `stderr`.
        result = e
    return result


def run_test_case(
    cli,
    module,
    function,
    plugin_name,
    backend,
    samples_path,
    fortran_driver,
    test_temp_dir,
    compiler="gfortran",
    extra_compiler_flags=(),
    expected_error_code=0,
    limited_area=False,
    env_vars=None,
):
    with cli.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(cli, module, function, plugin_name, backend, limited_area)
        compile_and_run_fortran(
            plugin_name,
            samples_path,
            fortran_driver,
            compiler,
            extra_compiler_flags,
            expected_error_code,
            env_vars,
            backend=backend,
        )


def invoke_cli(cli, module, function, plugin_name, backend, limited_area):
    cli_args = [module, function, plugin_name, "-b", backend, "-d"]
    if limited_area:
        cli_args.append("--limited-area")
    result = cli.invoke(main, cli_args)
    assert result.exit_code == 0, "CLI execution failed"


def compile_and_run_fortran(
    plugin_name,
    samples_path,
    fortran_driver,
    compiler,
    extra_compiler_flags,
    expected_error_code,
    env_vars,
    backend,
):
    try:
        compile_fortran_code(
            plugin_name, samples_path, fortran_driver, compiler, extra_compiler_flags, backend
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Compilation failed: {e}\n{e.stderr}\n{e.stdout}")

    try:
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        fortran_result = run_fortran_executable(plugin_name, env)
        if expected_error_code == 0:
            assert "passed" in fortran_result.stdout
        else:
            assert "failed" in fortran_result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")


@pytest.mark.parametrize(
    "run_backend, extra_flags",
    [
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
        ("CPU", ""),
    ],
)
def test_py2fgen_compilation_and_execution_square_cpu(
    cli_runner, run_backend, samples_path, square_wrapper_module, extra_flags, test_temp_dir
):
    """Tests embedding Python functions, and GT4Py program directly.
    Also tests embedding multiple functions in one shared library.
    """
    run_test_case(
        cli_runner,
        square_wrapper_module,
        "square,square_from_function",
        "square_plugin",
        run_backend,
        samples_path,
        "test_square",
        test_temp_dir,
        extra_compiler_flags=extra_flags,
    )


def test_py2fgen_python_error_propagation_to_fortran(
    cli_runner, samples_path, square_wrapper_module, test_temp_dir
):
    """Tests that Exceptions triggered in Python propagate an error code (1) up to Fortran."""
    run_test_case(
        cli_runner,
        square_wrapper_module,
        "square_error",
        "square_plugin",
        "ROUNDTRIP",
        samples_path,
        "test_square",
        test_temp_dir,
        extra_compiler_flags=("-DUSE_SQUARE_ERROR",),
        expected_error_code=1,
    )


@pytest.mark.skipif(os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI.")
@pytest.mark.parametrize(
    "function_name, plugin_name, test_name, run_backend, extra_flags",
    [
        ("square", "square_plugin", "test_square", "GPU", ("-acc", "-Minfo=acc")),
    ],
)
def test_py2fgen_compilation_and_execution_gpu(
    cli_runner,
    function_name,
    plugin_name,
    test_name,
    run_backend,
    samples_path,
    square_wrapper_module,
    extra_flags,
    test_temp_dir
):
    run_test_case(
        cli_runner,
        square_wrapper_module,
        function_name,
        plugin_name,
        run_backend,
        samples_path,
        test_name,
        test_temp_dir,
        os.environ["NVFORTRAN_COMPILER"],
        extra_compiler_flags=extra_flags,
        env_vars={"ICON4PY_BACKEND": "GPU"},
    )


@pytest.mark.parametrize(
    "run_backend, extra_flags",
    [
        ("CPU", ("-DPROFILE_SQUARE_FROM_FUNCTION",)),
    ],
)
def test_py2fgen_compilation_and_profiling(
    cli_runner, run_backend, samples_path, square_wrapper_module, extra_flags, test_temp_dir
):
    """Test profiling using cProfile of the generated wrapper."""
    run_test_case(
        cli_runner,
        square_wrapper_module,
        "square_from_function,profile_enable,profile_disable",
        "square_plugin",
        run_backend,
        samples_path,
        "test_square",
        test_temp_dir,
        extra_compiler_flags=extra_flags,
    )


@pytest.mark.skip("Need to adapt Fortran diffusion driver to pass connectivities.")
def test_py2fgen_compilation_and_execution_diffusion_gpu(cli_runner, samples_path, test_temp_dir):
    run_test_case(
        cli_runner,
        "icon4pytools.py2fgen.wrappers.diffusion",
        "diffusion_init,diffusion_run,profile_enable,profile_disable",
        "diffusion_plugin",
        "GPU",
        samples_path,
        "test_diffusion",
        test_temp_dir,
        os.environ["NVFORTRAN_COMPILER"],
        ("-acc", "-Minfo=acc"),
        limited_area=True,
        env_vars={"ICON4PY_BACKEND": "GPU"},
    )


@pytest.mark.skip("Need to adapt Fortran diffusion driver to pass connectivities.")
def test_py2fgen_compilation_and_execution_diffusion(cli_runner, samples_path, test_temp_dir):
    run_test_case(
        cli_runner,
        "icon4pytools.py2fgen.wrappers.diffusion",
        "diffusion_init,diffusion_run,profile_enable,profile_disable",
        "diffusion_plugin",
        "CPU",
        samples_path,
        "test_diffusion",
        test_temp_dir,
        limited_area=True,
    )
