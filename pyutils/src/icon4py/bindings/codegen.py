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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator, format_source
from functional.fencil_processors.codegens.gtfn.gtfn_backend import generate
from functional.ffront import program_ast as past
from functional.ffront.common_types import FieldType
from functional.iterator import ir as itir

from icon4py.bindings.cppgen import render_python_type
from icon4py.bindings.types import Field, Offset
from icon4py.bindings.utils import write_string
from icon4py.common.dimension import Koff
from icon4py.pyutils.exceptions import MultipleFieldOperatorException
from icon4py.pyutils.metadata import get_field_infos
from icon4py.pyutils.stencil_info import StencilInfo


class F90Generator(TemplatedGenerator):
    F90File = as_jinja(
        """
        #define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
        #define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
        module {{sten_name}}
        use, intrinsic :: iso_c_binding
        implicit none
        interface
        {{run_fun}}
        {{run_and_verify_fun}}
        {{setup_fun}}
        subroutine &
        free_{{sten_name}}( ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        end subroutine
        end interface
        contains
        {{wrap_run_fun}}
        {{wrap_setup_fun}}
        end module
    """
    )

    F90RunFun = as_jinja(
        """\
        subroutine &
        run_{{sten_name}}( &
        {% for field in _this_node.fields -%}
            {{field.name}}, &
        {% endfor -%}
        vertical_lower, &
        vertical_upper, &
        horizontal_lower, &
        horizontal_upper &
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {% for field in _this_node.fields -%}
            {{field.ctype()}}, dimension(*), target :: {{field.name}}
        {% endfor -%}
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        end subroutine
        """
    )

    F90RunAndVerifyFun = as_jinja(
        """
        subroutine &
        run_and_verify_{{sten_name}}( &
        {% for field in _this_node.all_fields -%}
            {{field.name}}, &
        {% endfor %}
        {%- for field in _this_node.out_fields -%}
            {{field.name}}_before, &
        {% endfor -%}
        vertical_lower, &
        vertical_upper, &
        horizontal_lower, &
        horizontal_upper, &
        {% for field in _this_node.out_fields -%}
        {{field.name}}_rel_tol, &
        {{field.name}}_abs_tol {% if not loop.last -%}
            , &
        {% else -%}
            &
        {%- endif %}
        {%- endfor %}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {% for field in _this_node.all_fields -%}
            {{field.ctype()}}, dimension(*), target :: {{field.name}}
        {% endfor %}
        {%- for field in _this_node.out_fields -%}
            {{field.ctype()}}, dimension(*), target :: {{field.name}}_before
        {% endfor -%}
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        {% for field in _this_node.out_fields -%}
        real(c_double), value, target :: {{field.name}}_rel_tol
        real(c_double), value, target :: {{field.name}}_abs_tol
        {%- endfor %}
        end subroutine
        """
    )

    F90WrapRunFun = as_jinja(
        """\
        subroutine &
        wrap_run_{{sten_name}}( &
        {% for field in _this_node.all_fields -%}
            {{field.name}}, &
        {% endfor %}
        {%- for field in _this_node.out_fields -%}
            {{field.name}}_before, &
        {% endfor -%}
        vertical_lower, &
        vertical_upper, &
        horizontal_lower, &
        horizontal_upper, &
        {% for field in _this_node.out_fields -%}
        {{field.name}}_rel_tol, &
        {{field.name}}_abs_tol{% if not loop.last -%}
            , &
        {% else %} &
        {%- endif %}
        {%- endfor %}
        )
        use, intrinsic :: iso_c_binding
        use openacc
        {% for field in _this_node.all_fields -%}
            {{field.ctype()}}, {{field.dim_string()}}, target :: {{field.name}}
        {% endfor -%}
        {% for field in _this_node.out_fields -%}
            {{field.ctype()}}, {{field.dim_string()}}, target :: {{field.name}}_before
        {% endfor -%}
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        {% for field in _this_node.out_fields -%}
        real(c_double), value, target, optional :: {{field.name}}_rel_tol
        real(c_double), value, target, optional :: {{field.name}}_abs_tol
        {%- endfor %}
        {% for field in _this_node.out_fields -%}
        real(c_double) :: {{field.name}}_rel_err_tol
        real(c_double) :: {{field.name}}_abs_err_tol
        {%- endfor %}
        integer(c_int) :: vertical_start
        integer(c_int) :: vertical_end
        integer(c_int) :: horizontal_start
        integer(c_int) :: horizontal_end
        vertical_start = vertical_lower-1
        vertical_end = vertical_upper
        horizontal_start = horizontal_lower-1
        horizontal_end = horizontal_upper
        {% for field in _this_node.out_fields -%}
        if (present({{field.name}}_rel_tol)) then
            {{field.name}}_rel_err_tol = {{field.name}}_rel_tol
        else
            {{field.name}}_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
        endif

        if (present({{field.name}}_abs_tol)) then
            {{field.name}}_abs_err_tol = {{field.name}}_abs_tol
        else
            {{field.name}}_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
        endif
        {%- endfor %}
        !$ACC host_data use_device( &
        {% for field in _this_node.all_fields -%}
        !$ACC {{field.name}}, &
        {% endfor %}
        {%- for field in _this_node.out_fields -%}
        !$ACC {{field.name}}_before, &
        {% endfor -%}
        !$ACC )
        #ifdef __DSL_VERIFY
            call run_and_verify_{{sten_name}} &
            ( &
            {% for field in _this_node.all_fields -%}
                {{field.name}}, &
            {% endfor %}
            {%- for field in _this_node.out_fields -%}
                {{field.name}}_before, &
            {% endfor -%}
            vertical_start, &
            vertical_end, &
            horizontal_start, &
            horizontal_end, &
            {% for field in _this_node.out_fields -%}
            {{field.name}}_rel_err_tol, &
            {{field.name}}_abs_err_tol{% if not loop.last -%}
            , &
            {% else %} &
            {% endif -%}
            {%- endfor -%}
            )
        #else
            call run_{{sten_name}} &
            ( &
            {% for field in _this_node.all_fields -%}
                {{field.name}}, &
            {% endfor %}
            vertical_start, &
            vertical_end, &
            horizontal_start, &
            horizontal_end &
            )
        #endif
        !$ACC end host_data
        end subroutine
        """
    )

    F90WrapSetupFun = as_jinja(
        """\
        subroutine &
        wrap_setup_{{sten_name}}( &
        mesh, &
        k_size, &
        stream, &
        {% for field in _this_node.out_fields -%}
            {{field.name}}_kmax{% if not loop.last -%}
            , &
            {% else %} &
            {%- endif -%}
        {% endfor %}
        )
        use, intrinsic :: iso_c_binding
        use openacc
        type(c_ptr), value, target :: mesh
        integer(c_int), value, target :: k_size
        integer(kind=acc_handle_kind), value, target :: stream
        {%- for field in _this_node.out_fields -%}
            integer(c_int), value, target, optional :: {{field.name}}_kmax
        {% endfor %}
        {%- for field in _this_node.out_fields -%}
            integer(c_int) :: {{field.name}}_kvert_max
        {% endfor %}
        {%- for field in _this_node.out_fields -%}

        if (present({{field.name}}_kmax)) then
            {{field.name}}_kvert_max = {{field.name}}_kmax
        else
            {{field.name}}_kvert_max = k_size
        endif
        {% endfor %}
        call setup_{{sten_name}} &
        ( &
            mesh, &
            k_size, &
            stream, &
            {% for field in _this_node.out_fields -%}
            {{field.name}}_kvert_max{% if not loop.last -%}
            , &
            {% else %} &
            {% endif -%}
            {%- endfor -%}
        )
        end subroutine
        """
    )

    F90SetupFun = as_jinja(
        """\
        subroutine &
        setup_{{sten_name}}( &
        mesh, &
        k_size, &
        stream, &
        {% for field in _this_node.out_fields -%}
            {{field.name}}_kmax{% if not loop.last -%}
            , &
        {%- else %} &
        {%- endif %}
        {%- endfor %}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        type(c_ptr), value, target :: mesh
        integer(c_int), value, target :: k_size
        integer(kind=acc_handle_kind), value, target :: stream
        {% for field in _this_node.out_fields -%}
            integer(c_int), value, target :: {{field.name}}_kmax {% if not loop.last -%}
            , &
        {% endif %}
        {%- endfor %}
        end subroutine
        """
    )


class F90RunFun(Node):
    sten_name: str
    fields: Sequence[Field]


class F90RunAndVerifyFun(Node):
    sten_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]


class F90SetupFun(Node):
    sten_name: str
    out_fields: Sequence[Field]


class F90WrapRunFun(Node):
    sten_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]


class F90WrapSetupFun(Node):
    sten_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]


class F90File(Node):
    sten_name: str
    run_fun: F90RunFun
    run_and_verify_fun: F90RunAndVerifyFun
    setup_fun: F90SetupFun
    wrap_run_fun: F90WrapRunFun
    wrap_setup_fun: F90WrapSetupFun


@dataclass
class F90Iface:
    sten_name: str
    fields: Sequence[Field]
    offsets: Sequence[Offset]

    def _generate_iface(self):
        iface = F90File(
            sten_name=self.sten_name,
            run_fun=F90RunFun(sten_name=self.sten_name, fields=self.fields),
            run_and_verify_fun=F90RunAndVerifyFun(
                sten_name=self.sten_name,
                all_fields=self.fields,
                out_fields=[field for field in self.fields if field.intent.out],
            ),
            setup_fun=F90SetupFun(
                sten_name=self.sten_name,
                out_fields=[field for field in self.fields if field.intent.out],
            ),
            wrap_run_fun=F90WrapRunFun(
                sten_name=self.sten_name,
                all_fields=self.fields,
                out_fields=[field for field in self.fields if field.intent.out],
            ),
            wrap_setup_fun=F90WrapSetupFun(
                sten_name=self.sten_name,
                all_fields=self.fields,
                out_fields=[field for field in self.fields if field.intent.out],
            ),
        )
        return iface

    def write(self, outpath: Path):
        iface = self._generate_iface()
        source = F90Generator.apply(iface)
        print(source)


class GTHeader:
    def __init__(self, stencil_info: StencilInfo):
        self.stencil_info = stencil_info
        self.stencil_name = stencil_info.fvprog.past_node.id

    def write(self, outpath: Path):
        gtheader = self._generate_cpp_code(
            self._adapt_domain(self.stencil_info.fvprog.itir)
        )
        write_string(gtheader, outpath, f"{self.stencil_name}.hpp")

    # TODO: provide a better typing for offset_provider
    def _generate_cpp_code(self, fencil: itir.FencilDefinition, **kwargs: Any) -> str:
        """Generate C++ code using the GTFN backend."""
        return generate(
            fencil,
            offset_provider=self.stencil_info.offset_provider,
            **kwargs,
        )

    def _adapt_domain(self, fencil: itir.FencilDefinition) -> itir.FencilDefinition:
        """Replace field view size parameters by horizontal and vertical range paramters."""
        if len(fencil.closures) > 1:
            raise MultipleFieldOperatorException()

        fencil.closures[0].domain = itir.FunCall(
            fun=itir.SymRef(id="unstructured_domain"),
            args=[
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value="horizontal"),
                        itir.SymRef(id="horizontal_start"),
                        itir.SymRef(id="horizontal_end"),
                    ],
                ),
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value=Koff.source.value),
                        itir.SymRef(id="vertical_start"),
                        itir.SymRef(id="vertical_end"),
                    ],
                ),
            ],
        )
        return itir.FencilDefinition(
            id=fencil.id,
            function_definitions=fencil.function_definitions,
            params=[
                *(p for p in fencil.params if not self._is_size_param(p)),
                itir.Sym(id="horizontal_start"),
                itir.Sym(id="horizontal_end"),
                itir.Sym(id="vertical_start"),
                itir.Sym(id="vertical_end"),
            ],
            closures=fencil.closures,
        )

    @staticmethod
    def _is_size_param(param: itir.Sym) -> bool:
        """Check if parameter is a size parameter introduced by field view frontend."""
        return param.id.startswith("__") and "_size_" in param.id


class CppHeader:
    def __init__(self, stencil_info: StencilInfo):
        self.stencil_info = stencil_info
        self.generator = HeaderGenerator
        self.fields = get_field_infos(stencil_info.fvprog)
        self.stencil_name = stencil_info.fvprog.past_node.id

    def write(self, outpath: Path):
        header = self._generate_header()
        source = format_source("cpp", self.generator.apply(header), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.h")

    def _generate_header(self):
        (
            all_params,
            before,
            k_size,
            out_dsl,
            tolerance,
        ) = self._get_template_data()

        header = HeaderFile(
            runFunc=StencilFuncDeclaration(
                funcname=self.stencil_name,
                parameters=all_params,
            ),
            verifyFunc=VerifyFuncDeclaration(
                funcname=self.stencil_name,
                out_dsl_params=out_dsl,
                tolerance_params=tolerance,
            ),
            runAndVerifyFunc=RunAndVerifyFunc(
                funcname=self.stencil_name,
                parameters=all_params,
                tolerance_params=tolerance,
                before_params=before,
            ),
            setupFunc=SetupFunc(funcname=self.stencil_name, parameters=k_size),
            freeFunc=FreeFunc(funcname=self.stencil_name),
        )
        return header

    def _get_template_data(self):
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

        if hasattr(f_info.field.field_type, "dims"):
            return True
        return False

    def _render_types(self, f_info, overload: Any = None):
        render_type = (
            self._handle_field_type(f_info.field) if not overload else overload
        )
        return render_python_type(np.dtype(render_type).field_type)

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
    # TODO: implement other header declarations
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
