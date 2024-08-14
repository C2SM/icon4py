# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from tempfile import TemporaryDirectory

from icon4pytools.liskov.codegen.shared.types import GeneratedCode
from icon4pytools.liskov.codegen.shared.write import DIRECTIVE_IDENT, CodegenWriter


def test_write_from():
    # create temporary directory and file
    with TemporaryDirectory() as temp_dir:
        input_filepath = Path(temp_dir) / "test.f90"
        output_filepath = input_filepath.with_suffix(".gen")

        with open(input_filepath, "w") as f:
            f.write("!$DSL\n some code\n another line")

        # create an instance of IntegrationWriter and write generated code
        generated = [GeneratedCode(1, "generated code")]
        integration_writer = CodegenWriter(input_filepath, output_filepath)
        integration_writer(generated)

        # check that the generated code was inserted into the file
        with open(output_filepath, "r") as f:
            content = f.read()
        assert "generated code" in content

        # check that the directive was removed from the file
        assert DIRECTIVE_IDENT not in content


def test_remove_directives():
    current_file = [
        "some code",
        "!$DSL directive",
        "another line",
        "!$DSL another directive",
    ]
    expected_output = ["some code", "another line"]
    assert CodegenWriter._remove_directives(current_file) == expected_output


def test_insert_generated_code():
    current_file = ["some code", "another line"]
    generated = [
        GeneratedCode(5, "generated code2"),
        GeneratedCode(1, "generated code1"),
    ]
    expected_output = [
        "some code",
        "generated code1\n",
        "another line",
        "generated code2\n",
    ]
    assert CodegenWriter._insert_generated_code(current_file, generated) == expected_output


def test_write_file():
    # create temporary directory and file
    with TemporaryDirectory() as temp_dir:
        input_filepath = Path(temp_dir) / "test.f90"
        output_filepath = input_filepath.with_suffix(".gen")

        generated_code = ["some code", "another line"]
        writer = CodegenWriter(input_filepath, output_filepath)
        writer._write_file(generated_code)

        # check that the generated code was written to the file
        with open(output_filepath, "r") as f:
            content = f.read()
        assert "some code" in content
        assert "another line" in content
