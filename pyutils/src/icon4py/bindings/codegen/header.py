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

from icon4py.bindings.types import Field
from icon4py.bindings.utils import write_string


@dataclass
class CppHeader:
    stencil_name: str
    fields: list[Field]

    def write(self, outpath: Path):
        header = self._generate_header()
        source = format_source("cpp", CppHeaderGenerator.apply(header), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.h")

    def _generate_header(self):
        output_fields = [field for field in self.fields if field.intent.out]
        header = CppHeaderFile(
            runFunc=CppRunFuncDeclaration(
                funcname=self.stencil_name,
                fields=self.fields,
            ),
            verifyFunc=CppVerifyFuncDeclaration(
                funcname=self.stencil_name,
                out_fields=output_fields,
            ),
            runAndVerifyFunc=CppRunAndVerifyFuncDeclaration(
                funcname=self.stencil_name, fields=self.fields, out_fields=output_fields
            ),
            setupFunc=CppSetupFunc(
                funcname=self.stencil_name, out_fields=output_fields
            ),
            freeFunc=CppFreeFunc(funcname=self.stencil_name),
        )
        return header


class CppFunc(Node):
    funcname: str


class CppFreeFunc(CppFunc):
    ...


class CppRunFuncDeclaration(CppFunc):
    fields: Sequence[Field]


class CppVerifyFuncDeclaration(CppFunc):
    out_fields: Sequence[Field]


class CppSetupFunc(CppVerifyFuncDeclaration):
    ...


class CppRunAndVerifyFuncDeclaration(CppRunFuncDeclaration, CppVerifyFuncDeclaration):
    ...


class CppHeaderFile(Node):
    runFunc: CppRunFuncDeclaration
    verifyFunc: CppVerifyFuncDeclaration
    runAndVerifyFunc: CppRunAndVerifyFuncDeclaration
    setupFunc: CppSetupFunc
    freeFunc: CppFreeFunc


run_func_declaration = as_jinja(
    """\
    void run_{{funcname}}(
    {%- for field in _this_node.fields -%}
    {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
    """
)

run_verify_func_declaration = as_jinja(
    """\
    void run_and_verify_{{funcname}}(
    {%- for field in _this_node.fields -%}
    {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
    {%- endfor -%}
    {%- for field in _this_node.out_fields -%}
    {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }}_before,
    {%- endfor -%}
    const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    {%- for field in _this_node.out_fields -%}
    const {{ field.ctype('c++')}} {{ field.name }}_rel_tol,
    const {{ field.ctype('c++')}} {{ field.name }}_abs_tol
    {%- if not loop.last -%}
    ,
    {%- endif -%}
    {%- endfor -%}
    ) ;
    """
)


class CppHeaderGenerator(TemplatedGenerator):
    CppHeaderFile = as_jinja(
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
    CppRunFuncDeclaration = run_func_declaration

    CppRunAndVerifyFuncDeclaration = run_verify_func_declaration

    CppVerifyFuncDeclaration = as_jinja(
        """\
        bool verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        const {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }}_dsl,
        const {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        const {{ field.ctype('c++')}} {{ field.name }}_rel_tol,
        const {{ field.ctype('c++')}} {{ field.name }}_abs_tol,
        {%- endfor -%}
        const int iteration) ;
        """
    )

    CppSetupFunc = as_jinja(
        """\
        void setup_{{funcname}}(
        dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_k_size
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%}) ;
        """
    )

    CppFreeFunc = as_jinja(
        """\
        void free_{{funcname}}() ;
        """
    )
