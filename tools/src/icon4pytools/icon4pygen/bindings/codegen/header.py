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
from typing import Any, Sequence

from gt4py import eve
from gt4py.eve import Node
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator, format_source

from icon4pytools.icon4pygen.bindings.entities import Field
from icon4pytools.icon4pygen.bindings.utils import write_string


run_func_declaration = as_jinja(
    """\
    void run_{{funcname}}(
    {%- for field in _this_node.fields -%}
    {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    {%- for field in _this_node.out_fields -%}
    const int {{ field.name }}_{{ k_size_suffix }},
    {%- endfor -%}
    const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd)
    """
)

run_verify_func_declaration = as_jinja(
    """\
    void run_and_verify_{{funcname}}(
    {%- for field in _this_node.fields -%}
    {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    {%- for field in _this_node.out_fields -%}
    {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_{{ before_suffix }},
    {%- endfor -%}
    {%- for field in _this_node.out_fields -%}
    const int {{ field.name }}_{{ k_size_suffix }},
    {%- endfor -%}
    {%- if _this_node.tol_fields -%}
    const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    {%- else -%}
    const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd
    {%- endif -%}
    {%- for field in _this_node.tol_fields -%}
    const double {{ field.name }}_rel_tol,
    const double {{ field.name }}_abs_tol
    {%- if not loop.last -%}
    ,
    {%- endif -%}
    {%- endfor -%}
    )
    """
)


class CppHeaderGenerator(TemplatedGenerator):
    CppHeaderFile = as_jinja(
        """\
        #pragma once
        #include "driver-includes/defs.hpp"
        #include "driver-includes/cuda_utils.hpp"
        extern "C" {
        {{ runFunc }};
        {{ verifyFunc }};
        {{ runAndVerifyFunc }};
        {{ setupFunc }};
        {{ freeFunc }};
        }
        """
    )
    CppRunFuncDeclaration = run_func_declaration

    CppRunAndVerifyFuncDeclaration = run_verify_func_declaration

    CppVerifyFuncDeclaration = as_jinja(
        """\
        bool verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_{{ before_suffix }},
        const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_{{ k_size_suffix }},
        {%- endfor -%}
        {%- for field in _this_node.tol_fields -%}
        const double {{ field.name }}_rel_tol,
        const double {{ field.name }}_abs_tol,
        {%- endfor -%}
        const int iteration)
        """
    )

    CppSetupFuncDeclaration = as_jinja(
        """\
        void setup_{{funcname}}(
        GlobalGpuTriMesh *mesh, cudaStream_t stream, json *json_record, verify *verify)
        """
    )

    CppFreeFunc = as_jinja(
        """\
        void free_{{funcname}}()
        """
    )


class CppFunc(Node):
    funcname: str


class CppFreeFunc(CppFunc):
    ...


class CppSizeFunc(CppFunc):
    out_fields: Sequence[Field]
    k_size_suffix: str


class CppRunFuncDeclaration(CppSizeFunc):
    fields: Sequence[Field]


class CppVerifyFuncDeclaration(CppSizeFunc):
    tol_fields: Sequence[Field]
    before_suffix: str


class CppSetupFuncDeclaration(CppFunc):
    ...


class CppRunAndVerifyFuncDeclaration(CppRunFuncDeclaration, CppVerifyFuncDeclaration):
    ...


class CppHeaderFile(Node):
    stencil_name: str
    fields: Sequence[Field]

    runFunc: CppRunFuncDeclaration = eve.datamodels.field(init=False)
    verifyFunc: CppVerifyFuncDeclaration = eve.datamodels.field(init=False)
    runAndVerifyFunc: CppRunAndVerifyFuncDeclaration = eve.datamodels.field(init=False)
    setupFunc: CppSetupFuncDeclaration = eve.datamodels.field(init=False)
    freeFunc: CppFreeFunc = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        output_fields = [field for field in self.fields if field.intent.out]
        tolerance_fields = [field for field in output_fields if not field.is_integral()]

        self.runFunc = CppRunFuncDeclaration(
            funcname=self.stencil_name,
            fields=self.fields,
            out_fields=output_fields,
            k_size_suffix="k_size",
        )

        self.verifyFunc = CppVerifyFuncDeclaration(
            funcname=self.stencil_name,
            out_fields=output_fields,
            tol_fields=tolerance_fields,
            before_suffix="dsl",
            k_size_suffix="k_size",
        )

        self.runAndVerifyFunc = CppRunAndVerifyFuncDeclaration(
            funcname=self.stencil_name,
            fields=self.fields,
            out_fields=output_fields,
            tol_fields=tolerance_fields,
            before_suffix="before",
            k_size_suffix="k_size",
        )

        self.setupFunc = CppSetupFuncDeclaration(
            funcname=self.stencil_name,
        )

        self.freeFunc = CppFreeFunc(funcname=self.stencil_name)


def generate_cpp_header(stencil_name: str, fields: list[Field], outpath: Path) -> None:
    header = CppHeaderFile(stencil_name=stencil_name, fields=fields)
    source = format_source("cpp", CppHeaderGenerator.apply(header), style="LLVM")
    write_string(source, outpath, f"{stencil_name}.h")
