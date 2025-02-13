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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common.dimension import (
    C2V,
    E2C,
    V2C2E,
    C2VDim,
    CellDim,
    EdgeDim,
    KDim,
    Koff,
    V2C2EDim,
    VertexDim,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_2nd_order_divergence_of_flux_of_normal_wind(
    geofac_2order_div: Field[[VertexDim, V2C2EDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    v2c_coeff: Field[[CellDim, C2VDim], wpfloat],
    dwdz: Field[[CellDim, KDim], wpfloat],
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
) -> tuple[Field[[EdgeDim, KDim], vpfloat].Field[[CellDim, KDim], vpfloat]]:
    div2order_vn_vertex = neighbor_sum(geofac_2order_div * vn(V2C2E), axis=V2C2EDim)
    div2order_vn = neighbor_sum(v2c_coeff(C2V) * div2order_vn_vertex(C2V), axis=C2VDim)
    div2order = dwdz + div2order_vn

    graddiv_normal = inv_dual_edge_length * div2order(E2C[1]) - div2order(E2C[0])
    graddiv_vertical = (div2order(Koff[-1]) - div2order) / ddqz_z_half

    return graddiv_normal, graddiv_vertical


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_2nd_order_divergence_of_flux_of_normal_wind(
    geofac_2order_div: Field[[VertexDim, V2C2EDim], wpfloat],
    # geofac_2order_div: Field[[VCEDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    z_flxdiv2order_vn_vertex: Field[[VertexDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_2nd_order_divergence_of_flux_of_normal_wind(
        geofac_2order_div,
        vn,
        out=z_flxdiv2order_vn_vertex,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
