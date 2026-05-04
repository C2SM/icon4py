# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_41."""
    z_flxdiv_mass_wp = neighbor_sum(
        geofac_div * mass_flux_at_edges_on_model_levels(C2E), axis=C2EDim
    )
    z_flxdiv_theta_wp = neighbor_sum(
        geofac_div * theta_v_flux_at_edges_on_model_levels(C2E), axis=C2EDim
    )
    return astype((z_flxdiv_mass_wp, z_flxdiv_theta_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[wpfloat],
    divergence_of_mass: fa.CellKField[vpfloat],
    divergence_of_theta_v: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div,
        mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels,
        out=(divergence_of_mass, divergence_of_theta_v),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
