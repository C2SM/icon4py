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

from icon4py.bindings.utils import format_fortran_code
from icon4py.liskov.codegen.integration import GeneratedCode
from icon4py.liskov.parsing.types import IDENTIFIER


class IntegrationWriter:
    SUFFIX = ".gen.f90"

    def __init__(self, generated: list[GeneratedCode]) -> None:
        self.generated = generated

    def write_from(self, filepath: Path) -> None:
        """Write a file containing generated code, with the DSL directives removed in the same directory as filepath."""
        with open(filepath, "r") as f:
            current_file = f.readlines()

        with_generated_code = self._insert_generated_code(current_file)
        without_directives = self._remove_directives(with_generated_code)

        self._to_file(filepath, without_directives)

    def _to_file(self, filepath: Path, generated_code: list[str]) -> None:
        """Format and write generated code to a file."""
        f = "\n".join(generated_code)
        formatted = format_fortran_code(f)
        new_file_path = filepath.with_suffix(self.SUFFIX)
        with open(new_file_path, "w") as f:
            f.write(formatted)

    def _insert_generated_code(self, current_file):
        # Keep track of the current line number in the current file
        cur_line_num = 0
        for gen in self.generated:
            # Update start and end line numbers in gen to account for any lines
            # that have been inserted into the current file so far
            gen.startln += cur_line_num

            to_insert = gen.source.split("\n")

            current_file[gen.startln : gen.startln] = to_insert

            # Update cur_line_num to account for any lines that have been inserted
            cur_line_num += len(to_insert)
        return current_file

    @staticmethod
    def _remove_directives(current_file):
        """Remove the directives."""
        return [ln for ln in current_file if IDENTIFIER not in ln]
