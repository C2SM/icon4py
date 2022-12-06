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

from dataclasses import asdict
from typing import Collection, Optional

import eve
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator

from icon4py.liskov.input import StencilData


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
    stencil_data: StencilData

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


# todo: Generation of in/out field declarations.
#   REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: vt_before
#   REAL(wp), DIMENSION(nproma,p_patch%nlevp1,p_patch%nblks_e) :: vn_ie_before
#   REAL(wp), DIMENSION(nproma,p_patch%nlev,p_patch%nblks_e) :: z_kin_hor_e_before


# todo: Copying of output fields.
#   !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
#   z_rth_pr_1_before(:,:,:) = z_rth_pr(:,:,:,1)
#   z_rth_pr_2_before(:,:,:) = z_rth_pr(:,:,:,2)
#   !$ACC END PARALLEL

# todo: Generation of profile call (requires adding command-line flag).
#   call nvtxStartRange("mo_solve_nonhydro_stencil_01")
#   ! Fortran stencil code goes here
#   call nvtxEndRange()

# todo: Generation of wrapped function call import statements (requires adding IMPORT directive).
#   USE mo_velocity_advection_stencil_01, ONLY: wrap_run_mo_velocity_advection_stencil_01

# todo: Generation/modification of DATA CREATE statement in Fortran code.
