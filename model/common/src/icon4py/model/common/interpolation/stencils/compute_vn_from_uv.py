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
def _compute_vn_from_uv(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Compute the velocity component normal to edges from the zonal and meridional
    wind components at cell centers.

    Port of ``compute_normal_velocity_edge`` in ICON's ``mo_vdf_atmo.f90``:

        vn(e, k) = sum over the two E2C neighbor cells c of
                   c_lin_e(e, c) * (u(c, k) * primal_normal_cell_x(e, c)
                                    + v(c, k) * primal_normal_cell_y(e, c))

    The Fortran call site (``Compute_diagnostics`` in ``mo_vdf_atmo.f90``) uses
    ``rl_start = grf_bdywidth_e + 1`` and ``rl_end = min_rledge_int``, i.e.
    ``h_grid.Zone.NUDGING_LEVEL_2`` to ``h_grid.Zone.LOCAL`` for edges, on all
    full levels.
    """
    return neighbor_sum(
        c_lin_e * (u(E2C) * primal_normal_cell_x + v(E2C) * primal_normal_cell_y),
        axis=E2CDim,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vn_from_uv(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vn_from_uv(
        u=u,
        v=v,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        c_lin_e=c_lin_e,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
