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

from icon4py.liskov.codegen.generate import GeneratedCode
from icon4py.liskov.codegen.write import DIRECTIVE_IDENT, IntegrationWriter


def test_write_from():
    # create temporary directory and file
    with TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test.f90"
        with open(filepath, "w") as f:
            f.write("!$DSL\n some code\n another line")

        # create an instance of IntegrationWriter with some generated code
        generated = [GeneratedCode("generated code", 1, 3)]
        integration_writer = IntegrationWriter(generated)

        # call the write_from method with the filepath
        integration_writer.write_from(filepath)

        # check that the generated code was inserted into the file
        with open(filepath.with_suffix(IntegrationWriter.SUFFIX), "r") as f:
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
        filepath = Path(temp_dir) / "test.f90"

        generated_code = ["some code", "another line"]
        writer = IntegrationWriter(generated_code)
        writer._write_file(filepath, generated_code)

        # check that the generated code was written to the file
        with open(filepath.with_suffix(IntegrationWriter.SUFFIX), "r") as f:
            content = f.read()
        assert "some code" in content
        assert "another line" in content
