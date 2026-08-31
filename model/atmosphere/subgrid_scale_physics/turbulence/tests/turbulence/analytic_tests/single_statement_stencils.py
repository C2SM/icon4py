# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""One '@gtx.program' per statement of a merged stencil, for the analytic tests.

The stencil merge (docs/superpowers/notes/2026-08-31-stencil-merge-plan.md) folds each section
of the scheme into one program with several statements, because a program that is nothing but a
list of field-operator calls is what a reader of the scheme recognises. The analytic tests need
the opposite: they run two statements as a FIXED-POINT ITERATION in the order the granule does
not use, and each of them is a MUTATION POINT that a defective copy from 'broken_stencils' is
substituted for. Neither is expressible against the merged program.

So the statements that the analytic tests drive are given a one-statement program here, over the
SAME field operator the merged program calls -- not a copy of its body. The arithmetic under
test is therefore the shipped arithmetic; what these wrappers add is a domain and a name.

The signatures are the ones the production programs had before they were merged, so the mutants
in 'broken_stencils' remain drop-in replacements.

WHAT THIS COSTS, stated rather than hidden: the analytic tests exercise the field operators of a
merged section and no longer a program the granule runs. The domains and the statement order of
the merged programs are covered by the section datatests in 'integration_tests/' instead.

A stencil argument list is positional by necessity -- the domain specification of a
'gtx.program' names its parameters -- which is why the project already exempts
'**/model/**/stencils/*.py' from PLR0917 in 'pyproject.toml'. These are stencils that happen to
live in the test tree, so they need the exemption spelled out here.
"""

# ruff: noqa: PLR0917 [too-many-positional-arguments]

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.solve_turb_budgets import (
    _compute_diffusion_coefficients_from_stability_lengths,
    _compute_stability_lengths,
    _compute_turbulent_velocity_scale,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


__all__ = [
    "compute_diffusion_coefficients_from_stability_lengths",
    "compute_stability_lengths",
    "compute_turbulent_velocity_scale",
]


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_diffusion_coefficients_from_stability_lengths(
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    molecular_diffusivity_for_scalars: wpfloat,
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """One statement of 'solve_turb_budgets', run alone so the round trip can be one.

    See 'ONE STATEMENT AT A TIME' above. This one is not a mutation point; it is separate
    because 'test_neutral_stability_functions.py' runs it against section 2c)'s
    'compute_stability_lengths_from_diffusion_coefficients' and asserts the two are inverse,
    which needs each of the pair on its own.
    """
    _compute_diffusion_coefficients_from_stability_lengths(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        turbulent_velocity_scale=turbulent_velocity_scale,
        molecular_diffusivity_for_scalars=molecular_diffusivity_for_scalars,
        out=(diffusion_coefficient_for_momentum, diffusion_coefficient_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_stability_lengths(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
    updated_stability_length_for_momentum: fa.CellKField[wpfloat],
    updated_stability_length_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """One statement of 'solve_turb_budgets', run alone so a mutation can be substituted for it.

    See 'ONE STATEMENT AT A TIME' above.
    """
    _compute_stability_lengths(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        turbulent_velocity_scale=turbulent_velocity_scale,
        a_h=a_h,
        a_m=a_m,
        b_h=b_h,
        b_m=b_m,
        d_m=d_m,
        d_1=d_1,
        d_2=d_2,
        d_3=d_3,
        d_4=d_4,
        d_5=d_5,
        d_6=d_6,
        rim=rim,
        frcsecu=frcsecu,
        stbsecu=stbsecu,
        out=(updated_stability_length_for_momentum, updated_stability_length_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_turbulent_velocity_scale(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """One statement of 'solve_turb_budgets', run alone so a mutation can be substituted for it.

    See 'ONE STATEMENT AT A TIME' above.
    """
    _compute_turbulent_velocity_scale(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        previous_velocity_scale=previous_velocity_scale,
        transport_tendency=transport_tendency,
        d_m=d_m,
        d_4=d_4,
        b_m=b_m,
        rim=rim,
        frcsecu=frcsecu,
        tkesecu=tkesecu,
        tkesmot=tkesmot,
        vel_min=vel_min,
        tke_time_step=tke_time_step,
        inverse_tke_time_step=inverse_tke_time_step,
        out=turbulent_velocity_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
