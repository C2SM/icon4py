# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import pathlib
import subprocess

import pytest
from click.testing import CliRunner

import icon4py.tools.py2fgen._utils as utils
from icon4py.tools.py2fgen._cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def square_wrapper_module():
    return "icon4py.bindings.simple"


def compile_fortran_code(
    library_name, samples_path, fortran_driver, compiler, extra_compiler_flags
):
    shared_library = f"{library_name}"
    command = [
        f"{compiler}",
        "-cpp",
        "-I.",
        "-Wl,-rpath=.",
        "-L.",
        f"{library_name}.f90",
        str(samples_path / f"{fortran_driver}.f90"),
        f"-l{shared_library}",
        "-o",
        library_name,
        *list(extra_compiler_flags),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def run_fortran_executable(library_name, env):
    try:
        result = subprocess.run(
            [f"./{library_name}"], capture_output=True, text=True, check=True, env=env
        )
    except subprocess.CalledProcessError as e:
        # If an error occurs, use the exception's `stdout` and `stderr`.
        result = e
    return result


def run_test_case(
    cli,
    module,
    function,
    library_name,
    samples_path,
    fortran_driver,
    test_temp_dir,
    compiler="gfortran",  # TODO(havogt): don't use hard-coded compiler, see gt4py.cartesian setuptools approach
    extra_compiler_flags=(),
    expected_error_code=0,
    env_vars=None,
):
    with cli.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(cli, module, function, library_name, extra_args=["--compile"])
        compile_and_run_fortran(
            library_name,
            samples_path,
            fortran_driver,
            compiler,
            extra_compiler_flags,
            expected_error_code,
            env_vars,
        )


def invoke_cli(cli, module, function, library_name, extra_args=None):
    rpath = utils.get_prefix_lib_path()

    cli_args = [module, function, library_name, "-r", rpath]
    if extra_args:
        cli_args.extend(extra_args)
    result = cli.invoke(main, cli_args)
    assert result.exit_code == 0, result.output
    return result


def compile_and_run_fortran(
    library_name,
    samples_path,
    fortran_driver,
    compiler,
    extra_compiler_flags,
    expected_error_code,
    env_vars,
):
    try:
        compile_fortran_code(
            library_name, samples_path, fortran_driver, compiler, extra_compiler_flags
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Compilation failed: {e}\n{e.stderr}\n{e.stdout}")

    try:
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        fortran_result = run_fortran_executable(library_name, env)
        if expected_error_code == 0:
            assert "passed" in fortran_result.stdout, fortran_result.stderr
        else:
            assert "failed" in fortran_result.stdout, fortran_result.stderr
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")


@pytest.mark.parametrize(
    "run_backend, extra_flags",
    [
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
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
        "square_from_function",
        "square_plugin",
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
        samples_path,
        "test_square",
        test_temp_dir,
        extra_compiler_flags=("-DUSE_SQUARE_ERROR",),
        expected_error_code=1,
    )


@pytest.mark.skipif(os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI.")
@pytest.mark.parametrize(
    "function_name, library_name, test_name, extra_flags",
    [
        (
            "square_from_function",
            "square_plugin",
            "test_square",
            ("-acc", "-Minfo=acc", "-DUSE_SQUARE_FROM_FUNCTION"),
        ),
    ],
)
def test_py2fgen_compilation_and_execution_gpu(
    cli_runner,
    function_name,
    library_name,
    test_name,
    samples_path,
    square_wrapper_module,
    extra_flags,
    test_temp_dir,
):
    run_test_case(
        cli_runner,
        square_wrapper_module,
        function_name,
        library_name,
        samples_path,
        test_name,
        test_temp_dir,
        os.environ["NVFORTRAN_COMPILER"],
        extra_compiler_flags=extra_flags,
        env_vars={"ICON4PY_BACKEND": "GPU"},
    )


@pytest.mark.parametrize(
    "extra_flags",
    [
        ("-DPROFILE_SQUARE_FROM_FUNCTION",),
    ],
)
def test_py2fgen_compilation_and_profiling(
    cli_runner, samples_path, square_wrapper_module, extra_flags, test_temp_dir, tmp_path
):
    """Test profiling using cProfile of the generated wrapper."""

    run_test_case(
        cli_runner,
        square_wrapper_module,
        "square_from_function",
        "square_plugin",
        samples_path,
        "test_square",
        test_temp_dir,
        extra_compiler_flags=extra_flags,
        env_vars={
            "PY2FGEN_EXTRA_CALLABLES": "icon4py.bindings.viztracer_plugin:init",
            "ICON4PY_TRACING_RANGE": "0:50",
            "ICON4PY_TRACING_NAMES": "square_from_function",
            "ICON4PY_TRACING_OUTPUT_DIR": str(tmp_path),
        },
    )
    assert (tmp_path / "viztracer.json").exists()


def test_py2fgen_incremental_skips_compilation_when_unchanged(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that running py2fgen --compile twice without changes skips compilation on the second run."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--compile"],
            )
            first_log = caplog.text
        assert "Compiling CFFI dynamic library" in first_log

        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--compile"],
            )
            second_log = caplog.text
        assert "Python wrapper is up to date" in second_log
        assert "Fortran interface is up to date" in second_log
        assert "Skipping compilation" in second_log
        assert "Compiling CFFI dynamic library" not in second_log


def test_py2fgen_regenerate_forces_recompilation(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that --regenerate forces recompilation even if files are up to date."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(
            cli_runner,
            square_wrapper_module,
            "square_from_function",
            "square_plugin",
            extra_args=["--compile"],
        )

        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--compile", "--regenerate"],
            )
            regen_log = caplog.text
        assert "Force regeneration requested" in regen_log
        assert "Compiling CFFI dynamic library" in regen_log
        assert "Skipping compilation" not in regen_log


def test_py2fgen_default_generates_sources_without_compiling(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Default invocation (no --compile) generates .py/.f90/.c/.h but no .so."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")
            log = caplog.text

        assert "Generating C source and header files" in log
        assert "Compiling CFFI dynamic library" not in log

        assert pathlib.Path("square_plugin.py").exists()
        assert pathlib.Path("square_plugin.f90").exists()
        assert pathlib.Path("square_plugin.h").exists()
        assert pathlib.Path("square_plugin.c").exists()
        assert not pathlib.Path("libsquare_plugin.so").exists()


def test_py2fgen_default_skips_when_up_to_date(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Default invocation (no --compile) skips C code regeneration when up to date."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")

        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")
            second_log = caplog.text
        assert "Skipping C code generation" in second_log
        assert "Generating C source and header files" not in second_log


def test_py2fgen_compile_with_output_flag_errors(cli_runner, square_wrapper_module, test_temp_dir):
    """``--compile`` combined with any --output-<kind> is rejected."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        rpath = utils.get_prefix_lib_path()
        result = cli_runner.invoke(
            main,
            [
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                "-r",
                rpath,
                "--compile",
                "--output-f90",
                "my.f90",
            ],
        )
        assert result.exit_code != 0
        assert "--compile cannot be combined with --output-" in result.output


def test_py2fgen_per_artifact_output_path(cli_runner, square_wrapper_module, test_temp_dir):
    """``--output-f90 PATH`` alone produces only that file at the custom path."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(
            cli_runner,
            square_wrapper_module,
            "square_from_function",
            "square_plugin",
            extra_args=["--output-f90", "my_fortran.f90"],
        )
        assert pathlib.Path("my_fortran.f90").exists()
        assert not pathlib.Path("square_plugin.f90").exists()
        assert not pathlib.Path("square_plugin.py").exists()
        assert not pathlib.Path("square_plugin.c").exists()
        assert not pathlib.Path("square_plugin.h").exists()
        assert not pathlib.Path("libsquare_plugin.so").exists()


def test_py2fgen_mixed_per_artifact_paths(cli_runner, square_wrapper_module, test_temp_dir):
    """``--output-f90`` and ``--output-c`` together emit both with custom names; .h goes alongside .c."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(
            cli_runner,
            square_wrapper_module,
            "square_from_function",
            "square_plugin",
            extra_args=[
                "--output-f90",
                "my_fortran.f90",
                "--output-c",
                "my_c.c",
            ],
        )
        assert pathlib.Path("my_fortran.f90").exists()
        assert pathlib.Path("my_c.c").exists()
        assert pathlib.Path("my_c.h").exists()
        assert not pathlib.Path("square_plugin.py").exists()
        assert not pathlib.Path("square_plugin.f90").exists()
        assert not pathlib.Path("square_plugin.c").exists()
        assert not pathlib.Path("square_plugin.h").exists()
        assert not pathlib.Path("libsquare_plugin.so").exists()


def test_py2fgen_output_h_override(cli_runner, square_wrapper_module, test_temp_dir):
    """``--output-h`` overrides the header path; the emitted .c references it via #include."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(
            cli_runner,
            square_wrapper_module,
            "square_from_function",
            "square_plugin",
            extra_args=[
                "--output-c",
                "foo.c",
                "--output-h",
                "custom.h",
            ],
        )
        assert pathlib.Path("foo.c").exists()
        assert pathlib.Path("custom.h").exists()
        assert not pathlib.Path("foo.h").exists()
        assert '#include "custom.h"' in pathlib.Path("foo.c").read_text()


def test_py2fgen_default_regenerates_if_c_file_deleted(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Default invocation regenerates .c if it is missing on the next run."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")
        assert pathlib.Path("square_plugin.c").exists()

        pathlib.Path("square_plugin.c").unlink()

        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")
            regen_log = caplog.text
        assert "Generating C source and header files" in regen_log
        assert pathlib.Path("square_plugin.c").exists()
