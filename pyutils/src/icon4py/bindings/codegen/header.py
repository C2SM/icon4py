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

import eve
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator, format_source

from icon4py.bindings.entities import Field
from icon4py.bindings.utils import write_string


DOMAIN_PARAMS = ["verticalStart", "verticalEnd", "horizontalStart", "horizontalEnd"]

domain_params = as_jinja(
    """\
    {%- for p in params -%}
    {% if typed == 'True' %} const int {%- endif %} {{ p }}
    {%- if not loop.last -%}
    ,
    {%- endif -%}
    {%- endfor -%}
    """
)

run_func_declaration = as_jinja(
    """\
    void run_{{funcname}}(
    {%- for field in _this_node.fields -%}
    {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    {{ domain_params }})
    """
)

run_verify_func_declaration = as_jinja(
    """\
    void run_and_verify_{{funcname}}(
    {%- for field in _this_node.fields -%}
    {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    {%- for field in _this_node.out_fields -%}
    {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_{{ suffix }},
    {%- endfor -%}
    {{ domain_params }},
    {%- for field in _this_node.out_fields -%}
    const double {{ field.name }}_rel_tol,
    const double {{ field.name }}_abs_tol
    {%- if not loop.last -%}
    ,
    {%- endif -%}
    {%- endfor -%}
    )
    """
)

cpp_verify_func_declaration = as_jinja(
    """\
    bool verify_{{funcname}}(
    {%- for field in _this_node.out_fields -%}
    const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_{{ suffix }},
    const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    {%- for field in _this_node.out_fields -%}
    const double {{ field.name }}_rel_tol,
    const double {{ field.name }}_abs_tol,
    {%- endfor -%}
    const int iteration)
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

    CppVerifyFuncDeclaration = cpp_verify_func_declaration

    CppSetupFuncDeclaration = as_jinja(
        """\
        void setup_{{funcname}}(
        dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_{{ suffix }}
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%})
        """
    )

    CppFreeFunc = as_jinja(
        """\
        void free_{{funcname}}()
        """
    )

    DomainParams = domain_params


class CppFunc(Node):
    funcname: str


class CppFreeFunc(CppFunc):
    ...


class DomainParams(Node):
    params: Sequence[str]
    typed: bool


class CppRunFuncDeclaration(CppFunc):
    fields: Sequence[Field]
    domain_params: DomainParams


class CppVerifyFuncDeclaration(CppFunc):
    out_fields: Sequence[Field]
    suffix: str


class CppSetupFuncDeclaration(CppVerifyFuncDeclaration):
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

    def __post_init__(self):
        output_fields = [field for field in self.fields if field.intent.out]

        self.runFunc = CppRunFuncDeclaration(
            funcname=self.stencil_name,
            fields=self.fields,
            domain_params=DomainParams(params=DOMAIN_PARAMS, typed=True),
        )

        self.verifyFunc = CppVerifyFuncDeclaration(
            funcname=self.stencil_name, out_fields=output_fields, suffix="dsl"
        )

        self.runAndVerifyFunc = CppRunAndVerifyFuncDeclaration(
            funcname=self.stencil_name,
            fields=self.fields,
            out_fields=output_fields,
            suffix="before",
            domain_params=DomainParams(params=DOMAIN_PARAMS, typed=True),
        )

        self.setupFunc = CppSetupFuncDeclaration(
            funcname=self.stencil_name, out_fields=output_fields, suffix="k_size"
        )

        self.freeFunc = CppFreeFunc(funcname=self.stencil_name)


def generate_cpp_header(stencil_name: str, fields: list[Field], outpath: Path):
    header = CppHeaderFile(stencil_name=stencil_name, fields=fields)
    source = format_source("cpp", CppHeaderGenerator.apply(header), style="LLVM")
    write_string(source, outpath, f"{stencil_name}.h")
