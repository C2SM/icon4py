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

import re
from dataclasses import asdict
from typing import Collection, Optional, Type

import eve
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator

from icon4py.bindings.utils import format_fortran_code
from icon4py.liskov.codegen.interface import (
    CodeGenInput,
    DeclareData,
    StartStencilData,
)


def generate_fortran_code(
    parent_node: Type[eve.Node],
    code_generator: Type[TemplatedGenerator],
    **kwargs: CodeGenInput | bool | list[str],
) -> str:
    parent = parent_node(**kwargs)
    source = code_generator.apply(parent)
    formatted_source = format_fortran_code(source)
    return formatted_source


class BoundsFields(eve.Node):
    vlower: str
    vupper: str
    hlower: str
    hupper: str


class Field(eve.Node):
    variable: str
    association: str
    abs_tol: Optional[str] = None
    rel_tol: Optional[str] = None
    inp: bool
    out: bool


class InputFields(eve.Node):
    fields: list[Field]


class OutputFields(InputFields):
    ...


class ToleranceFields(InputFields):
    ...


class WrapRunFunc(eve.Node):
    stencil_data: StartStencilData
    profile: bool

    name: str = eve.datamodels.field(init=False)
    input_fields: InputFields = eve.datamodels.field(init=False)
    output_fields: OutputFields = eve.datamodels.field(init=False)
    tolerance_fields: ToleranceFields = eve.datamodels.field(init=False)
    bounds_fields: BoundsFields = eve.datamodels.field(init=False)

    def __post_init__(self) -> None:  # type: ignore
        all_fields = [Field(**asdict(f)) for f in self.stencil_data.fields]
        self.bounds_fields = BoundsFields(**asdict(self.stencil_data.bounds))
        self.name = self.stencil_data.name
        self.input_fields = InputFields(fields=[f for f in all_fields if f.inp])
        self.output_fields = OutputFields(fields=[f for f in all_fields if f.out])
        self.tolerance_fields = ToleranceFields(
            fields=[f for f in all_fields if f.rel_tol or f.abs_tol]
        )


class WrapRunFuncGenerator(TemplatedGenerator):
    WrapRunFunc = as_jinja(
        """
        {%- if _this_node.profile %}
        call nvtxEndRange()
        {%- endif %}
        #endif
        call wrap_run_{{ name }}( &
            {{ input_fields }}
            {{ output_fields }}
            {{ tolerance_fields }}
            {{ bounds_fields }})
        """
    )

    InputFields = as_jinja(
        """
        {%- for field in _this_node.fields %}
            {{ field.variable }}={{ field.association }},&
        {%- endfor %}
        """
    )

    def visit_OutputFields(self, out: OutputFields) -> Collection[str] | str:
        f = out.fields[0]
        start_idx = f.association.find("(")
        end_idx = f.association.find(")")
        out_index = f.association[start_idx : end_idx + 1]

        return self.generic_visit(
            out, output_association=f"{f.variable}_before{out_index}"
        )

    OutputFields = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{ field.variable }}_before={{ output_association }},&
        {%- endfor -%}
        """
    )

    ToleranceFields = as_jinja(
        """
        {%- if _this_node.fields|length < 1 -%}

        {%- else -%}

            {%- for f in _this_node.fields -%}
                {%- if f.rel_tol -%}
                {{ f.variable }}_rel_tol={{ f.rel_tol }}, &
                {%- endif -%}
                {% if f.abs_tol %}
                {{ f.variable }}_abs_tol={{ f.abs_tol }}, &
                {% endif %}
            {%- endfor -%}

        {%- endif -%}
        """
    )

    BoundsFields = as_jinja(
        """vertical_lower={{ vlower }}, &
           vertical_upper={{ vupper }}, &
           horizontal_lower={{ hlower }}, &
           horizontal_upper={{ hupper }}
        """
    )


class Declaration(eve.Node):
    variable: str
    association: str


class DeclareStatement(eve.Node):
    declare_data: DeclareData
    declarations: list[Declaration] = eve.datamodels.field(init=False)

    def __post_init__(self) -> None:  # type: ignore
        self.declarations = [
            Declaration(variable=k, association=v)
            for dic in self.declare_data.declarations
            for k, v in dic.items()
        ]


class DeclareStatementGenerator(TemplatedGenerator):
    DeclareStatement = as_jinja(
        """
        !--------------------------------------------------------------------------
        ! OUT/INOUT FIELDS DSL
        !
        {%- for d in _this_node.declarations %}
        REAL(wp), DIMENSION({{ d.association }}) :: {{ d.variable }}_before
        {%- endfor %}
        """
    )


class CopyDeclaration(Declaration):
    array_index: str


class OutputFieldCopy(eve.Node):
    stencil_data: StartStencilData
    profile: bool
    copy_declarations: list[CopyDeclaration] = eve.datamodels.field(init=False)

    def __post_init__(self) -> None:  # type: ignore
        all_fields = [Field(**asdict(f)) for f in self.stencil_data.fields]
        out_fields = [
            Declaration(variable=f.variable, association=f.association)
            for f in all_fields
            if f.out
        ]
        self.copy_declarations = [self.make_copy_declaration(out) for out in out_fields]

    @staticmethod
    def make_copy_declaration(declr: Declaration) -> CopyDeclaration:
        dims = re.findall("\\(([^)]+)", declr.association)[0]
        copy_dim_params = list(dims)
        copy_dim_params[-1] = ":"
        copy_field_dims = f"({''.join(copy_dim_params)})"
        return CopyDeclaration(
            variable=declr.variable,
            association=declr.association,
            array_index=copy_field_dims,
        )


class OutputFieldCopyGenerator(TemplatedGenerator):
    OutputFieldCopy = as_jinja(
        """
        #ifdef __DSL_VERIFY
        !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
        {%- for d in _this_node.copy_declarations %}
        {{ d.variable }}_before{{ d.array_index }} = {{ d.variable }}{{ d.array_index }}
        {%- endfor %}
        !$ACC END PARALLEL

        {%- if _this_node.profile %}
        call nvtxStartRange("{{ _this_node.stencil_data.name }}")
        {%- endif %}
        """
    )


class ImportsStatement(eve.Node):
    names: list[str]


class ImportsStatementGenerator(TemplatedGenerator):
    ImportsStatement = as_jinja(
        """
        {%- for name in names %}
        USE {{ name }}, ONLY: wrap_run_{{ name }}
        {%- endfor %}
        """
    )
