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

from pathlib import Path
from tempfile import TemporaryDirectory

from icon4pytools.liskov.codegen.generate import GeneratedCode
from icon4pytools.liskov.codegen.write import DIRECTIVE_IDENT, IntegrationWriter


def test_write_from():
    # create temporary directory and file
    with TemporaryDirectory() as temp_dir:
        input_filepath = Path(temp_dir) / "test.f90"
        output_filepath = input_filepath.with_suffix(".gen")

        with open(input_filepath, "w") as f:
            f.write("!$DSL\n some code\n another line")

        # create an instance of IntegrationWriter and write generated code
        generated = [GeneratedCode("generated code", 1, 3)]
        integration_writer = IntegrationWriter(input_filepath, output_filepath)
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
    assert IntegrationWriter._remove_directives(current_file) == expected_output


def test_insert_generated_code():
    current_file = ["some code", "another line"]
    generated = [
        GeneratedCode("generated code2", 5, 6),
        GeneratedCode("generated code1", 1, 3),
    ]
    expected_output = [
        "some code",
        "generated code1\n",
        "another line",
        "generated code2\n",
    ]
    assert (
        IntegrationWriter._insert_generated_code(current_file, generated)
        == expected_output
    )


def test_write_file():
    # create temporary directory and file
    with TemporaryDirectory() as temp_dir:
        input_filepath = Path(temp_dir) / "test.f90"
        output_filepath = input_filepath.with_suffix(".gen")

        generated_code = ["some code", "another line"]
        writer = IntegrationWriter(input_filepath, output_filepath)
        writer._write_file(generated_code)

        # check that the generated code was written to the file
        with open(output_filepath, "r") as f:
            content = f.read()
        assert "some code" in content
        assert "another line" in content
