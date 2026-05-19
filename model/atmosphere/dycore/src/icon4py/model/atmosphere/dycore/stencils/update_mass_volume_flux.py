# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_mass_volume_flux(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    w: fa.CellKField[wpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_58."""
    z_a = r_nsubsteps * (
        vertical_mass_flux_at_cells_on_half_levels
        + rho_at_cells_on_half_levels * exner_w_implicit_weight_parameter * w
    )
    mass_flx_ic_wp = dynamical_vertical_mass_flux_at_cells_on_half_levels + z_a
    vol_flx_ic_wp = (
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels
        + z_a / rho_at_cells_on_half_levels
    )
    return mass_flx_ic_wp, vol_flx_ic_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_mass_volume_flux(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    w: fa.CellKField[wpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_mass_volume_flux(
        vertical_mass_flux_at_cells_on_half_levels,
        rho_at_cells_on_half_levels,
        exner_w_implicit_weight_parameter,
        w,
        dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        r_nsubsteps,
        out=(
            dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
