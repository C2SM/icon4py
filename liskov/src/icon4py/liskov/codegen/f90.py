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
from typing import Optional, Sequence, Type

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
    **kwargs: CodeGenInput | Sequence[CodeGenInput] | bool,
) -> str:
    """
    Generate Fortran code for the given parent node and code generator.

    Args:
        parent_node: A subclass of eve.Node that represents the parent node.
        code_generator: A subclass of TemplatedGenerator that will be used
            to generate the code.
        **kwargs: Arguments to be passed to the parent node constructor.
            This can be a single CodeGenInput value, a sequence of CodeGenInput
            values, or a boolean value, which is required by some parent nodes which
            require a profile argument.

    Returns:
        A string containing the formatted Fortran code.
    """
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


class EndStencilStatement(eve.Node):
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


class EndStencilStatementGenerator(TemplatedGenerator):
    EndStencilStatement = as_jinja(
        """
        {%- if _this_node.profile %}
        call nvtxEndRange()
        {%- endif %}
        #endif
        call wrap_run_{{ name }}( &
            {{ input_fields }}
            {{ output_fields }}
            {{ tolerance_fields }}
            {{ bounds_fields }}
        """
    )

    InputFields = as_jinja(
        """
        {%- for field in _this_node.fields %}
            {%- if field.out %}

            {%- else %}
            {{ field.variable }}={{ field.association }},&
            {%- endif -%}
        {%- endfor %}
        """
    )

    OutputFields = as_jinja(
        """
        {%- for field in _this_node.fields %}
            {{ field.variable }}={{ field.association }},&
            {{ field.variable }}_before={{ field.variable }}_before{{ field.out_index }},&
        {%- endfor %}
        """
    )

    def visit_OutputFields(self, out: OutputFields) -> OutputFields:  # type: ignore
        for f in out.fields:  # type: ignore
            start_idx = f.association.find("(")
            end_idx = f.association.find(")")
            out_index = f.association[start_idx : end_idx + 1]
            f.out_index = out_index
        return self.generic_visit(out)

    ToleranceFields = as_jinja(
        """
        {%- if _this_node.fields|length < 1 -%}

        {%- else -%}

            {%- for f in _this_node.fields -%}
                {% if f.rel_tol %}
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
           horizontal_upper={{ hupper }})
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
        ! DSL INPUT / OUTPUT FIELDS
        {%- for d in _this_node.declarations %}
        REAL(wp), DIMENSION({{ d.association }}) :: {{ d.variable }}_before
        {%- endfor %}
        LOGICAL :: dsl_verify
        """
    )


class CopyDeclaration(Declaration):
    array_index: str


class StartStencilStatement(eve.Node):
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
        old_dims = f"({dims})"
        new_association = declr.association.strip(old_dims)

        return CopyDeclaration(
            variable=declr.variable,
            association=new_association,
            array_index=copy_field_dims,
        )


class StartStencilStatementGenerator(TemplatedGenerator):
    StartStencilStatement = as_jinja(
        """
        #ifdef __DSL_VERIFY
        !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
        {%- for d in _this_node.copy_declarations %}
        {{ d.variable }}_before{{ d.array_index }} = {{ d.association }}{{ d.array_index }}
        {%- endfor %}
        !$ACC END PARALLEL

        {%- if _this_node.profile %}
        call nvtxStartRange("{{ _this_node.stencil_data.name }}")
        {%- endif %}
        """
    )


class ImportsStatement(eve.Node):
    stencils: list[StartStencilData]
    stencil_names: list[str] = eve.datamodels.field(init=False)

    def __post_init__(self) -> None:  # type: ignore
        self.stencil_names = [stencil.name for stencil in self.stencils]


class ImportsStatementGenerator(TemplatedGenerator):
    ImportsStatement = as_jinja(
        """  {% for name in stencil_names %}USE {{ name }}, ONLY: wrap_run_{{ name }}\n{% endfor %}"""
    )


class StartCreateStatement(eve.Node):
    stencils: list[StartStencilData]
    out_field_names: list[str] = eve.datamodels.field(init=False)

    def __post_init__(self) -> None:  # type: ignore
        self.out_field_names = [
            field.variable
            for stencil in self.stencils
            for field in stencil.fields
            if field.out
        ]


class StartCreateStatementGenerator(TemplatedGenerator):
    StartCreateStatement = as_jinja(
        """
        #ifdef __DSL_VERIFY
        dsl_verify = .TRUE.
        #elif
        dsl_verify = .FALSE.
        #endif

        !$ACC DATA CREATE( &
        {%- for name in out_field_names %}
        !$ACC   {{ name }}_before {%- if not loop.last -%}, & {% else %} & {%- endif -%}
        {%- endfor %}
        !$ACC   ), &
        !$ACC      IF ( i_am_accel_node .AND. acc_on .AND. dsl_verify)
        """
    )


class EndCreateStatement(eve.Node):
    ...


class EndCreateStatementGenerator(TemplatedGenerator):
    EndCreateStatement = as_jinja("!$ACC END DATA")
