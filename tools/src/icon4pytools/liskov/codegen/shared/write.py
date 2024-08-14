# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List

from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.shared.types import GeneratedCode
from icon4pytools.liskov.parsing.types import DIRECTIVE_IDENT
from icon4pytools.liskov.pipeline.definition import Step


logger = setup_logger(__name__)


class CodegenWriter(Step):
    def __init__(self, input_filepath: Path, output_filepath: Path) -> None:
        """Initialize an CodegenWriter instance.

        Args:
            input_filepath: Path to file containing directives.
            output_filepath: Path to file to write generated code.
        """
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def __call__(self, generated: List[GeneratedCode]) -> None:
        """Write a new file containing the generated code.

            Any !$DSL directives are removed from the file.

        Args:
            generated: A list of GeneratedCode instances representing the generated code that will be written to the file.
        """
        current_file = self._read_file()
        with_generated_code = self._insert_generated_code(current_file, generated)
        without_directives = self._remove_directives(with_generated_code)
        self._write_file(without_directives)

    def _read_file(self) -> List[str]:
        """Read the lines of the input file into a list.

        Returns:
            A list of strings representing the lines of the file.
        """
        with self.input_filepath.open("r") as f:
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

            to_insert = [f"{s}\n" for s in to_insert]

            current_file[gen.startln : gen.startln] = to_insert

            cur_line_num += len(to_insert)
        return current_file

    def _write_file(self, generated_code: List[str]) -> None:
        """Write generated code to a file.

        Args:
            generated_code: A list of strings representing the generated code to be written to the file.
        """
        code = "".join(generated_code)
        with self.output_filepath.open("w") as f:
            f.write(code)
        logger.info(f"Wrote new file to {self.output_filepath}")

    @staticmethod
    def _remove_directives(current_file: List[str]) -> List[str]:
        """Remove the directives from the current file.

        Args:
            current_file: A list of strings representing the lines of the current file.

        Returns:
            A list of strings representing the current file with the directives removed.
        """
        return [ln for ln in current_file if DIRECTIVE_IDENT not in ln]
