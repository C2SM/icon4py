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

import abc
from typing import Any, Optional, Sequence, Type

import gt4py.eve as eve
from gt4py.eve.codegen import TemplatedGenerator

from icon4pytools.icon4pygen.bindings.utils import format_fortran_code
from icon4pytools.liskov.codegen.shared.types import CodeGenInput, GeneratedCode
from icon4pytools.liskov.pipeline.definition import Step
from icon4pytools.liskov.codegen.integration.template import InsertStatement


class CodeGenerator(Step):
    def __init__(self) -> None:
        self.generated: list[GeneratedCode] = []

    @abc.abstractmethod
    def __call__(self, data: Any) -> list[GeneratedCode]:
        ...

    @staticmethod
    def _generate_fortran_code(
        parent_node: Type[eve.Node],
        code_generator: Type[TemplatedGenerator],
        **kwargs: CodeGenInput | Sequence[CodeGenInput] | Optional[bool],
    ) -> str:
        """
        Generate Fortran code for the given parent node and code generator.

        Args:
            parent_node: A subclass of eve.Node that represents the parent node.
            code_generator: A subclass of TemplatedGenerator that will be used
                to generate the code.
            **kwargs: Arguments to be passed to the parent node constructor.
                This can be a single CodeGenInput value, a sequence of CodeGenInput
                values, or a boolean value, which is required by some parent nodes which
                require a profile argument.

        Returns:
            A string containing the formatted Fortran code.
        """
        parent = parent_node(**kwargs)
        source = code_generator.apply(parent)
        # DSL INSERT Statements should be inserted verbatim meaning no fortran formatting
        if parent_node is InsertStatement:
          formatted_source = source
        else:
          formatted_source = format_fortran_code(source)
        return formatted_source

    def _generate(
        self,
        parent_node: Type[eve.Node],
        code_generator: Type[TemplatedGenerator],
        startln: int,
        **kwargs: CodeGenInput | Sequence[CodeGenInput] | Optional[bool] | Any,
    ) -> None:
        """Add a GeneratedCode object to the `generated` attribute with the given source code and line number information.

        Args:
            parent_node: The parent node of the code to be generated.
            code_generator: The code generator to use for generating the code.
            startln: The start line number of the generated code.
            **kwargs: Additional keyword arguments to be passed to the code generator.
        """
        source = self._generate_fortran_code(parent_node, code_generator, **kwargs)
        code = GeneratedCode(startln=startln, source=source)
        self.generated.append(code)
