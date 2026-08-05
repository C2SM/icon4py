# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_inverse_density_to_edges(
    rho: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Interpolate the air density from cell centers to edge midpoints and invert it.

    Port of the 'density at edge' block of 'Compute_diffusion_hor_wind'
    (mo_vdf.f90): 'cells2edges_scalar' with the linear E2C weights 'c_lin_e'
    followed by the in-place reciprocal loop:

        inv_rhoe = 1 / sum_{c in E2C} c_lin_e * rho(c)

    Vertical: all full levels (jk = 1..nlev).
    Horizontal (both the interpolation and the reciprocal loop):
    rl_start = grf_bdywidth_e + 1 -> 'h_grid.Zone.NUDGING_LEVEL_2' (edges),
    rl_end = min_rledge_int -> 'h_grid.Zone.LOCAL' (edges).

    Args:
        rho: air density at cell centers on full levels [kg/m^3]
        c_lin_e: cell-to-edge linear interpolation coefficients

    Returns:
        inverse air density at edge midpoints on full levels [m^3/kg]
    """
    return wpfloat("1.0") / neighbor_sum(rho(E2C) * c_lin_e, axis=E2CDim)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_inverse_density_to_edges(
    rho: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_inverse_density_to_edges(
        rho=rho,
        c_lin_e=c_lin_e,
        out=inv_rhoe,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
