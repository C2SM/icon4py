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
import tempfile
from pathlib import Path

import numpy as np
import pytest
from cffi import FFI

from icon4pytools.py2fgen.plugin import generate_and_compile_cffi_plugin, unpack


@pytest.fixture
def ffi():
    return FFI()


@pytest.mark.parametrize(
    "data, expected_result",
    [
        ([1.0, 2.0, 3.0, 4.0], np.array([[1.0, 3.0], [2.0, 4.0]])),
        ([1, 2, 3, 4], np.array([[1, 3], [2, 4]])),
    ],
)
def test_unpack_column_major(data, expected_result, ffi):
    ptr = ffi.new("double[]", data) if isinstance(data[0], float) else ffi.new("int[]", data)

    rows, cols = expected_result.shape

    result = unpack(ptr, rows, cols)

    assert np.array_equal(result, expected_result)


def test_compile_and_run_cffi_plugin_from_C():
    plugin_name = "test_plugin"
    c_header = "int test_function();"
    c_source_code = (
        f"""
    #include <stdio.h>
    #include \"{plugin_name}.h\"
    """
        + """
    int main() {
        printf(\"%d\\n\", test_function());
        return 0;
    }
    """
    )

    python_wrapper = """
    from test_plugin import ffi

    @ffi.def_extern()
    def test_function():
        return 42
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        build_path = Path(tmpdirname)

        try:
            # Generate and compile the CFFI plugin, which creates lib{plugin_name}.so
            generate_and_compile_cffi_plugin(plugin_name, c_header, python_wrapper, build_path)
            compiled_library_path = build_path / f"lib{plugin_name}.so"

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
                    build_path / "test_program",
                    str(main_program_path),
                    "-L" + str(build_path),
                    "-l" + plugin_name,
                    "-Wl,-rpath=" + str(build_path),
                ],
                check=True,
            )

            # Execute the compiled program and capture its output
            result = subprocess.run(
                [build_path / "test_program"], stdout=subprocess.PIPE, check=True
            )
            output = result.stdout.decode().strip()

            # Assert the output of test_function called within the C program
            assert output == "42", f"Expected '42', got '{output}'"

        except Exception as e:
            pytest.fail(
                f"Unexpected error during plugin generation, compilation, or execution: {e}"
            )
