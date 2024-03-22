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
from typing import Any, Collection, Optional

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator

from icon4pytools.liskov.codegen.integration.exceptions import UndeclaredFieldError
from icon4pytools.liskov.codegen.integration.interface import (
    BaseStartStencilData,
    DeclareData,
    StartFusedStencilData,
    StartStencilData,
)
from icon4pytools.liskov.external.metadata import CodeMetadata


def enclose_in_parentheses(string: str) -> str:
    return f"({string})"


class BoundsFields(eve.Node):
    vlower: str
    vupper: str
    hlower: str
    hupper: str


class Assign(eve.Node):
    variable: str
    association: str


class Field(Assign):
    dims: Optional[int]
    abs_tol: Optional[str] = None
    rel_tol: Optional[str] = None
    inp: bool
    out: bool


class InputFields(eve.Node):
    fields: list[Field]


class OutputFields(InputFields):
    verification: bool


class ToleranceFields(InputFields):
    ...


def get_array_dims(association: str) -> str:
    """
    Return the dimensions of an array in a string format.

    Args:
        association: The string representation of the array.
    """
    indexes = re.findall("\\(([^)]+)", association)
    if len(indexes) > 1:
        idx = indexes[-1]
    else:
        idx = indexes[0]

    dims = list(idx)

    return "".join(list(dims))


class MetadataStatement(eve.Node):
    metadata: CodeMetadata


class MetadataStatementGenerator(TemplatedGenerator):
    MetadataStatement = as_jinja(
        """\
    !+-+-+-+-+-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+
    ! GENERATED WITH ICON-LISKOV
    !+-+-+-+-+-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+
    ! Generated on: {{ _this_node.metadata.generated_on }}
    ! Input filepath: {{ _this_node.metadata.cli_params['input_path'] }}
    ! Profiling active: {{ _this_node.metadata.cli_params['profile'] }}
    ! Git version tag: {{ _this_node.metadata.version }}
    !+-+-+-+-+-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+
    """
    )


class Declaration(Assign):
    ...


class CopyDeclaration(Declaration):
    lh_index: str
    rh_index: str


def _make_copy_declaration(f: Field) -> CopyDeclaration:
    if f.dims is None:
        raise UndeclaredFieldError(f"{f.variable} was not declared!")

    lh_idx = render_index(f.dims)

    # get length of association index
    association_dims = get_array_dims(f.association).split(",")
    n_association_dims = len(association_dims)

    offset = len(",".join(association_dims)) + 2
    truncated_association = f.association[:-offset]

    if n_association_dims > f.dims:
        rh_idx = f"{lh_idx},{association_dims[-1]}"
    else:
        rh_idx = f"{lh_idx}"

    lh_idx = enclose_in_parentheses(lh_idx)
    rh_idx = enclose_in_parentheses(rh_idx)

    return CopyDeclaration(
        variable=f.variable,
        association=truncated_association,
        lh_index=lh_idx,
        rh_index=rh_idx,
    )


class DeclareStatement(eve.Node):
    declare_data: DeclareData
    declarations: list[Declaration] = eve.datamodels.field(init=False)
    verification: bool

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.declarations = [
            Declaration(variable=k, association=v)
            for k, v in self.declare_data.declarations.items()
        ]


class DeclareStatementGenerator(TemplatedGenerator):
    DeclareStatement = as_jinja(
        """
        ! DSL INPUT / OUTPUT FIELDS
        {%- for d in _this_node.declarations %}
        {%- if (_this_node.verification or _this_node.declare_data.suffix!='before' ) %}
        {{ _this_node.declare_data.ident_type }}, DIMENSION({{ d.association }}) :: {{ d.variable }}_{{ _this_node.declare_data.suffix }}
        {%- endif -%}
        {%- endfor %}
        """
    )




class EndBasicStencilStatement(eve.Node):
    name: str = eve.datamodels.field(init=False)
    input_fields: InputFields = eve.datamodels.field(init=False)
    output_fields: OutputFields = eve.datamodels.field(init=False)
    tolerance_fields: ToleranceFields = eve.datamodels.field(init=False)
    bounds_fields: BoundsFields = eve.datamodels.field(init=False)
    verification: bool


class EndStencilStatement(EndBasicStencilStatement):
    stencil_data: StartStencilData
    profile: bool
    noendif: Optional[bool]
    noprofile: Optional[bool]
    noaccenddata: Optional[bool]

    def __post_init__(self, *args, **kwargs) -> None:
        all_fields = [Field(**asdict(f)) for f in self.stencil_data.fields]
        self.bounds_fields = BoundsFields(**asdict(self.stencil_data.bounds))
        self.name = self.stencil_data.name
        self.input_fields = InputFields(fields=[f for f in all_fields if f.inp])
        self.output_fields = OutputFields(fields=[f for f in all_fields if f.out], verification=self.verification)
        if self.verification:
          self.tolerance_fields = ToleranceFields(
              fields=[f for f in all_fields if f.rel_tol or f.abs_tol],
          )
        else:
          self.tolerance_fields = ToleranceFields(fields=[])

class EndFusedStencilStatement(EndBasicStencilStatement):
    stencil_data: StartFusedStencilData
    copy_declarations: list[CopyDeclaration] = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        all_fields = [Field(**asdict(f)) for f in self.stencil_data.fields]
        self.copy_declarations = [_make_copy_declaration(f) for f in all_fields if f.out]
        self.bounds_fields = BoundsFields(**asdict(self.stencil_data.bounds))
        self.name = self.stencil_data.name
        self.input_fields = InputFields(fields=[f for f in all_fields if f.inp])
        self.output_fields = OutputFields(fields=[f for f in all_fields if f.out], verification=self.verification)
        if self.verification:
          self.tolerance_fields = ToleranceFields(
              fields=[f for f in all_fields if f.rel_tol or f.abs_tol], verification=self.verification
          )
        else:
          self.tolerance_fields = ToleranceFields(fields=[])



class BaseEndStencilStatementGenerator(TemplatedGenerator):
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

    def visit_OutputFields(self, out: OutputFields) -> str | Collection[str]:
        for f in out.fields:
            if f.dims is None:
                continue

            idx = render_index(f.dims)
            split_idx = idx.split(",")

            if len(split_idx) >= 3:
                split_idx[-1] = "1"

            setattr(f, "rh_index", enclose_in_parentheses(",".join(split_idx)))
        return self.generic_visit(out)

    OutputFields = as_jinja(
        """
        {%- for field in _this_node.fields %}
            {{ field.variable }}={{ field.association }},&
            {%- if _this_node.verification %}
              {{ field.variable }}_before={{ field.variable }}_before{{ field.rh_index }},&
            {%- endif -%}
        {%- endfor %}
        """
    )

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


class EndStencilStatementGenerator(BaseEndStencilStatementGenerator):
    EndStencilStatement = as_jinja(
        """
        {%- if _this_node.profile %}
        {% if _this_node.noprofile %}{% else %}call nvtxEndRange(){% endif %}
        {%- endif %}
        {% if _this_node.noendif %}{% else %}#endif{% endif %}
        {%- if _this_node.verification %}
        call wrap_run_and_verify_{{ name }}( &
        {% else %} 
        call wrap_run_{{ name }}( &
        {%- endif -%}
            {{ input_fields }}
            {{ output_fields }}
            {{ tolerance_fields }}
            {{ bounds_fields }}

        {%- if _this_node.verification %}
        {%- if not _this_node.noaccenddata %}
        !$ACC END DATA
        {%- endif %}
        {%- endif -%}
        """
    )


class EndFusedStencilStatementGenerator(BaseEndStencilStatementGenerator):
    EndFusedStencilStatement = as_jinja(
        """
        {%- if _this_node.verification %}
        call wrap_run_and_verify_{{ name }}( &
        {% else %} 
        call wrap_run_{{ name }}( &
        {%- endif -%}
            {{ input_fields }}
            {{ output_fields }}
            {{ tolerance_fields }}
            {{ bounds_fields }}

        {%- if _this_node.verification %}
        !$ACC EXIT DATA DELETE( &
        {%- for d in _this_node.copy_declarations %}
        !$ACC   {{ d.variable }}_before {%- if not loop.last -%}, & {% else %} ) & {%- endif -%}
        {%- endfor %}
        !$ACC      IF ( i_am_accel_node )
        {%- endif %}
        """
    )


class StartStencilStatement(eve.Node):
    stencil_data: StartStencilData
    profile: bool
    verification: bool
    copy_declarations: list[CopyDeclaration] = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        all_fields = [Field(**asdict(f)) for f in self.stencil_data.fields]
        self.copy_declarations = [_make_copy_declaration(f) for f in all_fields if f.out]
        self.acc_present = "PRESENT" if self.stencil_data.acc_present else "NONE"


class StartFusedStencilStatement(eve.Node):
    stencil_data: StartFusedStencilData
    copy_declarations: list[CopyDeclaration] = eve.datamodels.field(init=False)
    verification: bool

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        all_fields = [Field(**asdict(f)) for f in self.stencil_data.fields]
        self.copy_declarations = [_make_copy_declaration(f) for f in all_fields if f.out]
        self.acc_present = "PRESENT" if self.stencil_data.acc_present else "NONE"


def render_index(n: int) -> str:
    """
    Render a string of comma-separated colon characters, used to define the shape of an array in Fortran.

    Args:
        n (int): The number of colons to include in the returned string.

    Returns:
        str: A comma-separated string of n colons.

    Example:
        >>> render_index(3)
        ':,:,:'
    """
    return ",".join([":" for _ in range(n)])


class StartStencilStatementGenerator(TemplatedGenerator):
    StartStencilStatement = as_jinja(
        """

        {%- if _this_node.verification %}
        !$ACC DATA CREATE( &
        {%- for d in _this_node.copy_declarations %}
        !$ACC   {{ d.variable }}_before {%- if not loop.last -%}, & {% else %} ) & {%- endif -%}
        {%- endfor %}
        !$ACC      IF ( i_am_accel_node )
        {%- endif -%}

        #ifdef __DSL_VERIFY
        {%- if _this_node.verification %}
        {% if _this_node.stencil_data.copies -%}
        !$ACC KERNELS IF( i_am_accel_node ) DEFAULT({{ _this_node.acc_present }}) ASYNC(1)
        {%- for d in _this_node.copy_declarations %}
        {{ d.variable }}_before{{ d.lh_index }} = {{ d.association }}{{ d.rh_index }}
        {%- endfor %}
        !$ACC END KERNELS
        {%- endif -%}
        {%- endif -%}

        {%- if _this_node.profile %}
        call nvtxStartRange("{{ _this_node.stencil_data.name }}")
        {%- endif %}
        """
    )


class StartFusedStencilStatementGenerator(TemplatedGenerator):
    StartFusedStencilStatement = as_jinja(
        """

        {%- if _this_node.verification %}
        !$ACC ENTER DATA CREATE( &
        {%- for d in _this_node.copy_declarations %}
        !$ACC   {{ d.variable }}_before {%- if not loop.last -%}, & {% else %} ) & {%- endif -%}
        {%- endfor %}
        !$ACC      IF ( i_am_accel_node )

        #ifdef __DSL_VERIFY
        !$ACC KERNELS IF( i_am_accel_node ) DEFAULT(PRESENT) ASYNC(1)
        {%- for d in _this_node.copy_declarations %}
        {{ d.variable }}_before{{ d.lh_index }} = {{ d.association }}{{ d.rh_index }}
        {%- endfor %}
        !$ACC END KERNELS
        #endif
        {%- endif -%}

        """
    )


class ImportsStatement(eve.Node):
    stencils: list[BaseStartStencilData]
    stencil_names: list[str] = eve.datamodels.field(init=False)
    verification: bool

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.stencil_names = sorted(set([stencil.name for stencil in self.stencils]))


class ImportsStatementGenerator(TemplatedGenerator):
    ImportsStatement = as_jinja(
        """
        {% for name in stencil_names %}
        {%- if _this_node.verification %}
        USE {{ name }}, ONLY: wrap_run_and_verify_{{ name }}\n
        {% else %}
        USE {{ name }}, ONLY: wrap_run_{{ name }}\n
        {%- endif -%}
        {% endfor %}"""
    )


class StartCreateStatement(eve.Node):
    extra_fields: Optional[list[str]]


class StartCreateStatementGenerator(TemplatedGenerator):
    StartCreateStatement = as_jinja(
        """
        !$ACC DATA CREATE( &
        {%- if _this_node.extra_fields -%}
        {%- for name in extra_fields %}
        !$ACC   {{ name }} {%- if not loop.last -%}, & {% else %} ) & {%- endif -%}
        {%- endfor %}
        {%- endif %}
        !$ACC   IF ( i_am_accel_node )
        """
    )


class EndCreateStatement(eve.Node):
    ...


class EndCreateStatementGenerator(TemplatedGenerator):
    EndCreateStatement = as_jinja("!$ACC END DATA")


class StartDeleteStatement(eve.Node):
    ...


class StartDeleteStatementGenerator(TemplatedGenerator):
    StartDeleteStatement = as_jinja("#ifdef __DSL_VERIFY")


class EndDeleteStatement(eve.Node):
    ...


class EndDeleteStatementGenerator(TemplatedGenerator):
    EndDeleteStatement = as_jinja("#endif")


class EndIfStatement(eve.Node):
    ...


class EndIfStatementGenerator(TemplatedGenerator):
    EndIfStatement = as_jinja("#endif")


class StartProfileStatement(eve.Node):
    name: str


class StartProfileStatementGenerator(TemplatedGenerator):
    StartProfileStatement = as_jinja('call nvtxStartRange("{{ _this_node.name }}")')


class EndProfileStatement(eve.Node):
    ...


class EndProfileStatementGenerator(TemplatedGenerator):
    EndProfileStatement = as_jinja("call nvtxEndRange()")


class InsertStatement(eve.Node):
    content: str


class InsertStatementGenerator(TemplatedGenerator):
    InsertStatement = as_jinja("{{ _this_node.content }}")
