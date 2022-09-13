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
from typing import Sequence

import numpy as np
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator

from icon4py.bindings.cppgen import render_python_type
from icon4py.pyutils.metadata import get_field_infos
from icon4py.pyutils.stencil_info import StencilInfo


class CppHeader:
    def __init__(self, stencil_info: StencilInfo):
        self.stencil_info = stencil_info
        self.generator = HeaderGenerator
        self.fields = get_field_infos(stencil_info.fvprog)

    def write(self, outpath: Path):
        header = self._generate_header()
        source = self.generator.apply(header)
        self._source_to_file(outpath, source)

    def _generate_header(self):
        stencil_name = self.stencil_info.fvprog.past_node.id
        dsl_params = self._make_verify_params("_dsl")
        rel_params = self._make_verify_params("_rel_tol")
        abs_params = self._make_verify_params("_abs_tol")
        out_params = self._make_verify_params()

        header = HeaderFile(
            runFunc=StencilFuncDeclaration(
                funcname=stencil_name,
                parameters=[
                    FunctionParameter(
                        name=field_name,
                        dtype=np.dtype(field_info.field.type.dtype.__str__()),
                        inp=field_info.inp,
                        out=field_info.out,
                    )
                    for field_name, field_info in self.fields.items()
                ],
            ),
            # TODO: return correct parameters (dsl fields, output fields, rel/abs fields)
            verifyFunc=VerifyFuncDeclaration(
                funcname=stencil_name,
                dsl_params=dsl_params,
                rel_params=rel_params,
                abs_params=abs_params,
                out_params=out_params,
            ),
        )
        return header

    def _make_verify_params(self, pstring=""):

        names = [
            str(f"{f}{pstring}") for f, f_info in self.fields.items() if f_info.out
        ]
        types = [
            np.dtype(f_info.field.type.dtype.__str__())
            for f, f_info in self.fields.items()
            if f_info.out
        ]

        field_dict = dict(zip(names, types))

        return [
            FunctionParameter(name=name, dtype=dtype, inp=False, out=True)
            for name, dtype in field_dict.items()
        ]

    def _source_to_file(self, outpath: Path, src: str):
        # write even if dir does not exist
        header_path = outpath / f"{self.stencil_info.fvprog.past_node.id}.h"
        with open(header_path, "w") as f:
            f.write(src)


class FunctionParameter(Node):
    name: str
    dtype: np.dtype
    inp: bool
    out: bool


class StencilFuncDeclaration(Node):
    funcname: str
    parameters: Sequence[FunctionParameter]


class VerifyFuncDeclaration(Node):
    # TODO: add (dsl fields, output fields, rel/abs fields)
    funcname: str
    dsl_params: Sequence[FunctionParameter]
    out_params: Sequence[FunctionParameter]
    rel_params: Sequence[FunctionParameter]
    abs_params: Sequence[FunctionParameter]


class HeaderFile(Node):
    runFunc: StencilFuncDeclaration
    verifyFunc: VerifyFuncDeclaration


class HeaderGenerator(TemplatedGenerator):
    # TODO: implement other header declarations
    HeaderFile = as_jinja(
        """\
        #include "driver-includes/defs.hpp"
        #include "driver-includes/cuda_utils.hpp"
        extern "C" {
        {{runFunc}}
        {{verifyFunc}}
        }
        """
    )

    StencilFuncDeclaration = as_jinja(
        """\
        void run_{{funcname}}({{",".join(parameters)}}, const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
        """
    )

    VerifyFuncDeclaration = as_jinja(
        """\
        bool verify_{{funcname}}({{",".join(dsl_params)}}, {{",".join(out_params)}}, {{",".join(abs_params)}}, {{",".join(rel_params)}}, const int iteration) ;
        """
    )

    def visit_FunctionParameter(self, param: FunctionParameter):
        type_str = render_python_type(param.dtype.type)
        p = f"{type_str} *{param.name}"
        return p
