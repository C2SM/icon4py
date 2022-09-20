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

from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator, format_source

from icon4py.bindings.types import Field, Offset
from icon4py.bindings.utils import write_string


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
            {{field.ctype('f90')}}, dimension(*), target :: {{field.name}}
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
            {{field.ctype('f90')}}, dimension(*), target :: {{field.name}}
        {% endfor %}
        {%- for field in _this_node.out_fields -%}
            {{field.ctype('f90')}}, dimension(*), target :: {{field.name}}_before
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
            {{field.ctype('f90')}}, {{field.dim_string()}}, target :: {{field.name}}
        {% endfor -%}
        {% for field in _this_node.out_fields -%}
            {{field.ctype('f90')}}, {{field.dim_string()}}, target :: {{field.name}}_before
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
        # todo: code formatting & writing code to file.
        print(source)


class CppHeader:
    def __init__(
        self,
        stencil_name: str,
        fields: list[Field],
    ):
        self.stencil_name = stencil_name
        self.fields = fields

    def write(self, outpath: Path):
        header = self._generate_header()
        source = format_source("cpp", HeaderGenerator.apply(header), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.h")

    def _generate_header(self):
        output_fields = [field for field in self.fields if field.intent.out]
        header = HeaderFile(
            runFunc=StencilFuncDeclaration(
                funcname=self.stencil_name,
                fields=self.fields,
            ),
            verifyFunc=VerifyFuncDeclaration(
                funcname=self.stencil_name,
                out_fields=output_fields,
            ),
            runAndVerifyFunc=RunAndVerifyFunc(
                funcname=self.stencil_name, fields=self.fields, out_fields=output_fields
            ),
            setupFunc=SetupFunc(funcname=self.stencil_name, out_fields=output_fields),
            freeFunc=FreeFunc(funcname=self.stencil_name),
        )
        return header


class Func(Node):
    funcname: str


class FreeFunc(Func):
    ...


class StencilFuncDeclaration(Func):
    fields: Sequence[Field]


class VerifyFuncDeclaration(Func):
    out_fields: Sequence[Field]


class SetupFunc(VerifyFuncDeclaration):
    ...


class RunAndVerifyFunc(StencilFuncDeclaration, VerifyFuncDeclaration):
    ...


class HeaderFile(Node):
    runFunc: StencilFuncDeclaration
    verifyFunc: VerifyFuncDeclaration
    runAndVerifyFunc: RunAndVerifyFunc
    setupFunc: SetupFunc
    freeFunc: FreeFunc


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
        void run_{{funcname}}(
        {% for field in _this_node.fields -%}
        {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
        {% endfor -%}
        const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
        """
    )

    VerifyFuncDeclaration = as_jinja(
        """\
        bool verify_{{funcname}}(
        {% for field in _this_node.out_fields -%}
        const {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }}_dsl,
        const {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
        {% endfor -%}
        {% for field in _this_node.out_fields -%}
        const {{ field.ctype('c++')}} {{ field.name }}_rel_tol,
        const {{ field.ctype('c++')}} {{ field.name }}_abs_tol,
        {% endfor -%}
        const int iteration) ;
        """
    )

    RunAndVerifyFunc = as_jinja(
        """\
        void run_and_verify_{{funcname}}(
        {% for field in _this_node.fields -%}
        {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
        {% endfor -%}
        {% for field in _this_node.out_fields -%}
        {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }}_before,
        {% endfor -%}
        const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd,
        {% for field in _this_node.out_fields -%}
        const {{ field.ctype('c++')}} {{ field.name }}_rel_tol,
        const {{ field.ctype('c++')}} {{ field.name }}_abs_tol
        {% if not loop.last -%}
        ,
        {% endif -%}
        {% endfor -%}
        ) ;
        """
    )

    SetupFunc = as_jinja(
        """\
        void setup_{{funcname}}(
        dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
        {% for field in _this_node.out_fields -%}
        const int {{ field.name }}_k_size
        {% if not loop.last -%}
        ,
        {% endif -%}
        {% endfor -%}) ;
        """
    )

    FreeFunc = as_jinja(
        """\
        void free_{{funcname}}() ;
        """
    )
