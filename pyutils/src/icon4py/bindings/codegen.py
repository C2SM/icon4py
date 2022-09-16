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

import itertools
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator, format_source
from functional.ffront import program_ast as past
from functional.ffront.common_types import FieldType

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
        source = format_source("cpp", self.generator.apply(header), style="LLVM")
        self._source_to_file(outpath, source)

    def _generate_header(self):
        (
            all_params,
            before,
            k_size,
            out_dsl,
            tolerance,
            stencil_name,
        ) = self._get_template_data()

        header = HeaderFile(
            runFunc=StencilFuncDeclaration(
                funcname=stencil_name,
                parameters=all_params,
            ),
            verifyFunc=VerifyFuncDeclaration(
                funcname=stencil_name,
                out_dsl_params=out_dsl,
                tolerance_params=tolerance,
            ),
            runAndVerifyFunc=RunAndVerifyFunc(
                funcname=stencil_name,
                parameters=all_params,
                tolerance_params=tolerance,
                before_params=before,
            ),
            setupFunc=SetupFunc(funcname=stencil_name, parameters=k_size),
            freeFunc=FreeFunc(funcname=stencil_name),
        )
        return header

    def _get_template_data(self):
        stencil_name = self.stencil_info.fvprog.past_node.id
        output = self._make_output_params("", is_const=True)
        all_params = self._make_output_params("", select_all=True)
        k_size = self._make_output_params(
            "_k_size", is_const=True, type_overload=np.intc
        )
        out_dsl = self._interleave_params(
            self._make_output_params("_dsl", is_const=True), output
        )
        tolerance = self._interleave_params(
            self._make_output_params("_rel_tol", is_const=True),
            self._make_output_params("_abs_tol", is_const=True),
        )
        before = self._make_output_params("_before", is_const=False)
        return (
            all_params,
            before,
            k_size,
            out_dsl,
            tolerance,
            stencil_name,
        )

    @staticmethod
    def _interleave_params(*args):
        return list(itertools.chain(*zip(*args)))

    def _make_output_params(
        self,
        pname: str,
        is_const: bool = False,
        type_overload: Any = None,
        select_all: bool = False,
    ):

        params = []

        for f, f_info in self.fields.items():

            if select_all:
                is_out = f_info.out
            else:
                is_out = True

            if f_info.out == is_out:
                name = f"{f}{pname}"
                dtype = self._render_types(f_info, type_overload)
                is_pointer = self._is_pointer(f_info, name)
                params.append(
                    FunctionParameter(
                        name=name,
                        dtype=dtype,
                        out=is_out,
                        pointer=is_pointer,
                        const=is_const,
                    )
                )
        return params

    @staticmethod
    def _is_pointer(f_info, name):
        for f in ["_abs_tol", "_rel_tol", "_k_size"]:
            if f in name:
                return False

        if hasattr(f_info.field.type, "dims"):
            return True
        return False

    def _render_types(self, f_info, overload: Any = None):
        render_type = (
            self._handle_field_type(f_info.field) if not overload else overload
        )
        return render_python_type(np.dtype(render_type).type)

    def _source_to_file(self, outpath: Path, src: str):
        header_path = outpath / f"{self.stencil_info.fvprog.past_node.id}.h"
        with open(header_path, "w") as f:
            f.write(src)

    @staticmethod
    def _handle_field_type(field: past.DataSymbol):
        if isinstance(field.type, FieldType):
            return str(field.type.dtype)
        return str(field.type)


class Func(Node):
    funcname: str


class FunctionParameter(Node):
    name: str
    dtype: str
    out: bool
    pointer: bool
    const: bool


class StencilFuncDeclaration(Func):
    parameters: Sequence[FunctionParameter]


class VerifyFuncDeclaration(Func):
    out_dsl_params: Sequence[FunctionParameter]
    tolerance_params: Sequence[FunctionParameter]


class RunAndVerifyFunc(StencilFuncDeclaration):
    tolerance_params: Sequence[FunctionParameter]
    before_params: Sequence[FunctionParameter]


class SetupFunc(StencilFuncDeclaration):
    ...


class FreeFunc(Func):
    ...


class HeaderFile(Node):
    runFunc: StencilFuncDeclaration
    verifyFunc: VerifyFuncDeclaration
    runAndVerifyFunc: RunAndVerifyFunc
    setupFunc: StencilFuncDeclaration
    freeFunc: Func


class HeaderGenerator(TemplatedGenerator):
    HeaderFile = as_jinja(
        """\
        #pragma once
        #include "driver-includes/defs.hpp"
        #include "driver-includes/cuda_utils.hpp"
        extern "C" {
        {{ runFunc }}
        {{ verifyFunc }}
        {{ runAndVerifyFunc }}
        {{ setupFunc }}
        {{ freeFunc }}
        }
        """
    )

    StencilFuncDeclaration = as_jinja(
        """\
        void run_{{funcname}}({{", ".join(parameters)}}, const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
        """
    )

    VerifyFuncDeclaration = as_jinja(
        """\
        bool verify_{{funcname}}({{", ".join(out_dsl_params)}}, {{", ".join(tolerance_params)}}, const int iteration) ;
        """
    )

    RunAndVerifyFunc = as_jinja(
        """\
        void run_and_verify_{{funcname}}({{", ".join(parameters)}}, {{", ".join(before_params)}}, const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd, {{", ".join(tolerance_params)}}) ;
        """
    )

    SetupFunc = as_jinja(
        """\
        void setup_{{funcname}}(dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream, {{", ".join(parameters)}}) ;
        """
    )

    FreeFunc = as_jinja(
        """\
        void free_{{funcname}}() ;
        """
    )

    def visit_FunctionParameter(self, param: FunctionParameter):
        const = "const " if param.const else ""
        pointer = "*" if param.pointer else ""
        type_str = f"{const}{param.dtype}{pointer} {param.name}"
        return type_str
