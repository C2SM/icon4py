# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _update_mass_flux_weighted(
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_65."""
    w_concorr_c_wp = astype(contravariant_correction_at_cells_on_half_levels, wpfloat)

    mass_flx_ic_wp = dynamical_vertical_mass_flux_at_cells_on_half_levels + (
        r_nsubsteps * rho_at_cells_on_half_levels * (exner_w_explicit_weight_parameter * w_now + exner_w_implicit_weight_parameter * w_new - w_concorr_c_wp)
    )
    return mass_flx_ic_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_mass_flux_weighted(
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_mass_flux_weighted(
        rho_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter,
        exner_w_implicit_weight_parameter,
        w_now,
        w_new,
        contravariant_correction_at_cells_on_half_levels,
        dynamical_vertical_mass_flux_at_cells_on_half_levels,
        r_nsubsteps,
        out=dynamical_vertical_mass_flux_at_cells_on_half_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
