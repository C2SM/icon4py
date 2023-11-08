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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (  # noqa: A004 # import gt4py builtin
    Field,
    abs,
    broadcast,
    int32,
    minimum,
    neighbor_sum,
    where,
)

from icon4py.model.common.dimension import (
    E2C,
    E2C2EO,
    E2V,
    CellDim,
    E2C2EODim,
    E2CDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
)


@field_operator
def _mo_velocity_advection_stencil_20(
    levelmask: Field[[KDim], bool],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    area_edge: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    zeta: Field[[VertexDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    vn: Field[[EdgeDim, KDim], float],
    ddt_vn_apc: Field[[EdgeDim, KDim], float],
    cfl_w_limit: float,
    scalfac_exdiff: float,
    dtime: float,
) -> Field[[EdgeDim, KDim], float]:
    w_con_e = broadcast(0.0, (EdgeDim, KDim))
    difcoef = broadcast(0.0, (EdgeDim, KDim))

    w_con_e = where(
        levelmask | levelmask(Koff[1]),
        neighbor_sum(c_lin_e * z_w_con_c_full(E2C), axis=E2CDim),
        w_con_e,
    )
    difcoef = where(
        (levelmask | levelmask(Koff[1])) & (abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        scalfac_exdiff
        * minimum(
            0.85 - cfl_w_limit * dtime,
            abs(w_con_e) * dtime / ddqz_z_full_e - cfl_w_limit * dtime,
        ),
        difcoef,
    )
    ddt_vn_apc = where(
        (levelmask | levelmask(Koff[1])) & (abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        ddt_vn_apc
        + difcoef
        * area_edge
        * (
            neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim)
            + tangent_orientation * inv_primal_edge_length * (zeta(E2V[1]) - zeta(E2V[0]))
        ),
        ddt_vn_apc,
    )
    return ddt_vn_apc


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_20(
    levelmask: Field[[KDim], bool],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    area_edge: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    zeta: Field[[VertexDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    vn: Field[[EdgeDim, KDim], float],
    ddt_vn_apc: Field[[EdgeDim, KDim], float],
    cfl_w_limit: float,
    scalfac_exdiff: float,
    dtime: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_velocity_advection_stencil_20(
        levelmask,
        c_lin_e,
        z_w_con_c_full,
        ddqz_z_full_e,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        zeta,
        geofac_grdiv,
        vn,
        ddt_vn_apc,
        cfl_w_limit,
        scalfac_exdiff,
        dtime,
        out=ddt_vn_apc,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
