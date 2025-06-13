# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import subprocess
import tempfile

import pytest

from icon4py.tools.py2fgen._codegen import BindingsLibrary
from icon4py.tools.py2fgen._generator import generate_and_compile_cffi_plugin, get_cffi_description

from . import utils


def test_parse_functions_on_wrapper():
    # TODO make independent of `wrappers`
    module_path = "icon4py.tools.py2fgen.wrappers.diffusion_wrapper"
    functions = ["diffusion_init", "diffusion_run"]
    plugin = get_cffi_description(module_path, functions, "diffusion_plugin")
    assert isinstance(plugin, BindingsLibrary)


def test_compile_and_run_cffi_plugin_from_C():
    rpath = utils.get_prefix_lib_path()
    print(rpath)
    library_name = "test_plugin"
    c_header = "int test_function();"
    c_source_code = f"""
    #include <stdio.h>
    #include "{library_name}.h"

    int main() {{
        printf("%d\\n", test_function());
        return 0;
    }}
    """

    python_wrapper = """
    from test_plugin import ffi

    @ffi.def_extern()
    def test_function():
        return 42
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        build_path = pathlib.Path(tmpdirname)

        try:
            # Generate and compile the CFFI plugin, which creates lib{library_name}.so
            shared_library = f"{library_name}"

            generate_and_compile_cffi_plugin(
                library_name, c_header, python_wrapper, build_path, rpath
            )
            compiled_library_path = build_path / f"lib{shared_library}.so"

            # Verify the shared library was created
            assert (
                compiled_library_path.exists()
            ), f"Compiled library {compiled_library_path} does not exist."
            assert compiled_library_path.stat().st_size > 0, "Compiled library is empty."

            # Write the main C program to a file
            main_program_path = build_path / "main.c"
            with open(main_program_path, "w") as main_program_file:
                main_program_file.write(c_source_code)

            # Compile the main C program against the shared library
            subprocess.run(
                [
                    "gcc",
                    "-o",
                    str(build_path / "test_program"),
                    str(main_program_path),
                    "-L" + str(build_path),
                    "-l" + shared_library,
                    "-Wl,-rpath=" + str(build_path),
                ],
                check=True,
                capture_output=True,
            )

            # Execute the compiled program and capture its output
            result = subprocess.run(
                [build_path / "test_program"], stdout=subprocess.PIPE, check=True
            )
            output = result.stdout.decode().strip()

            # Assert the output of test_function called within the C program
            assert output == "42", f"Expected '42', got '{output}'"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")
        except Exception as e:
            pytest.fail(
                f"Unexpected error during plugin generation, compilation, or execution: {e}"
            )
