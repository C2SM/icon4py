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
from eve.codegen import MakoTemplate as as_mako
from eve.codegen import TemplatedGenerator, format_source

from icon4py.bindings.cppgen import render_python_type
from icon4py.pyutils.stencil_info import StencilInfo
from icon4py.bindings.types import Field
from icon4py.bindings.types import Offset


class F90Generator(TemplatedGenerator):
    F90Fields = as_jinja(
        """\
        {% for field in fields %}
            {{field}}
        {% endfor %}
        """
    )

    def visit_Field(self, field: Field):
        return f"real({field.type}), dimension(*), target :: {field.name}"


class F90Fields(Node):
    fields: Sequence[Field]


@dataclass
class F90Iface:
    fields: Sequence[Field]
    offsets: Sequence[Offset]

    def _generate_iface(self):
        iface = F90Fields(fields=self.fields)
        return iface

    def write(self, outpath: Path):
        iface = self._generate_iface()
        source = F90Generator.apply(iface)
        print(source)


@dataclass
class CppHeader:
    stencil_info: StencilInfo

    def write(self, outpath: Path):
        header = self._generate_header()
        source = format_source("cpp", HeaderGenerator.apply(header), style="LLVM")
        self._source_to_file(outpath, source)

    def _generate_header(self):
        header = HeaderFile(
            runFunc=StencilFuncDeclaration(
                funcname=self.stencil_info.fvprog.past_node.id,
                parameters=[
                    FunctionParameter(
                        name=param.id, dtype=np.dtype(param.type.dtype.__str__())
                    )
                    for param in self.stencil_info.fvprog.past_node.params
                ],
            )
        )
        return header

    def _source_to_file(self, outpath: Path, src: str):
        # write even if dir does not exist
        header_path = outpath / f"{self.stencil_info.fvprog.past_node.id}.h"
        with open(header_path, "w") as f:
            f.write(src)


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
        void run_{{funcname}}({{",".join(parameters)}}, const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
        """
    )

    def visit_FunctionParameter(self, param: FunctionParameter):
        type_str = render_python_type(param.dtype.type)
        p = f"{type_str} *{param.name}"
        return p
