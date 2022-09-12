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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator, format_source

from icon4py.bindings.cppgen import render_python_type
from icon4py.pyutils.stencil_info import StencilInfo


@dataclass
class CppHeader:
    stencil_info: StencilInfo

    def write(self, output_path: Path):
        header = self._generate_header()
        src = format_source("cpp", HeaderGenerator.apply(header), style="LLVM")
        self.string_to_file(header, output_path, src)

    def _generate_header(self):
        # TODO: read attributes form stencil info (the below is a test).
        header = HeaderFile(
            runFunc=StencilFuncDeclaration(
                funcname="foo",
                parameters=[
                    FunctionParameter(name="bar", dtype=np.dtype(np.int32)),
                    FunctionParameter(name="bar2", dtype=np.dtype(np.float32)),
                    FunctionParameter(name="bar3", dtype=np.dtype(np.bool)),
                ],
            )
        )
        return header

    def _make_header_file_path(self, output_path, src):
        ...


class FunctionParameter(Node):
    name: str
    dtype: np.dtype


class StencilFuncDeclaration(Node):
    funcname: str
    parameters: Sequence[FunctionParameter]


class HeaderFile(Node):
    runFunc: StencilFuncDeclaration


class HeaderGenerator(TemplatedGenerator):
    # TODO: implement other header declarations
    HeaderFile = as_jinja(
        """\
        #include "driver-includes/defs.hpp"
        #include "driver-includes/cuda_utils.hpp"
        extern "C" {
        {{runFunc}}
        }
        """
    )

    StencilFuncDeclaration = as_jinja(
        """\
        void run_{{funcname}}({{", ".join(parameters)}}, const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
        """
    )

    def visit_FunctionParameter(self, param: FunctionParameter):
        type_str = render_python_type(param.dtype.type)
        return type_str + " " + param.name
