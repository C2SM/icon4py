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
from typing import List

from icon4py.bindings.utils import format_fortran_code
from icon4py.liskov.codegen.generate import GeneratedCode
from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.types import DIRECTIVE_IDENT


logger = setup_logger(__name__)


class IntegrationWriter:
    SUFFIX = ".gen.f90"

    def __init__(self, generated: List[GeneratedCode]) -> None:
        """Initialize an IntegrationWriter instance with a list of generated code.

        Args:
            generated: A list of GeneratedCode instances representing the generated code that will be written to a file.
        """
        self.generated = generated

    def write_from(self, filepath: Path) -> None:
        """Write a file containing generated code, with the DSL directives removed in the same directory as filepath using a new suffix.

        Args:
            filepath: Path to file containing directives.
        """
        current_file = self._read_file(filepath)
        with_generated_code = self._insert_generated_code(current_file, self.generated)
        without_directives = self._remove_directives(with_generated_code)
        self._write_file(filepath, without_directives)

    @staticmethod
    def _read_file(filepath: Path) -> List[str]:
        """Read the lines of a file into a list.

        Args:
            filepath: Path to the file to be read.

        Returns:
            A list of strings representing the lines of the file.
        """
        with filepath.open("r") as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def _insert_generated_code(
        current_file: List[str], generated_code: List[GeneratedCode]
    ) -> List[str]:
        """Insert generated code into the current file at the specified line numbers.

        The generated code is sorted in ascending order of the start line number to ensure that
        it is inserted into the current file in the correct order. The `cur_line_num` variable is
        used to keep track of the current line number in the current file, and is updated after
        each generated code block is inserted to account for any additional lines that have been
        added to the file.

        Args:
            current_file: A list of strings representing the lines of the current file.
            generated_code: A list of GeneratedCode instances representing the generated code to be inserted into the current file.

        Returns:
            A list of strings representing the current file with the generated code inserted at the
            specified line numbers.
        """
        generated_code.sort(key=lambda gen: gen.startln)
        cur_line_num = 0

        for gen in generated_code:
            gen.startln += cur_line_num

            to_insert = gen.source.split("\n")

            current_file[gen.startln : gen.startln] = to_insert

            cur_line_num += len(to_insert)
        return current_file

    @staticmethod
    def _write_file(filepath: Path, generated_code: List[str]) -> None:
        """Format and write generated code to a file.

        Args:
            filepath: Path to the file where the generated code will be written.
            generated_code: A list of strings representing the generated code to be written to the file.
        """
        code = "\n".join(generated_code)
        formatted_code = format_fortran_code(code)
        new_file_path = filepath.with_suffix(IntegrationWriter.SUFFIX)
        with new_file_path.open("w") as f:
            f.write(formatted_code)
        logger.info(f"Wrote new file to {new_file_path}")

    @staticmethod
    def _remove_directives(current_file: List[str]) -> List[str]:
        """Remove the directives from the current file.

        Args:
            current_file: A list of strings representing the lines of the current file.

        Returns:
            A list of strings representing the current file with the directives removed.
        """
        return [ln for ln in current_file if DIRECTIVE_IDENT not in ln]
