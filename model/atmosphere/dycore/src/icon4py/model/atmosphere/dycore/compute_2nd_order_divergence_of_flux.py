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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common.dimension import (
    C2CE,
    C2E,
    C2V,
    V2C,
    C2EDim,
    C2VDim,
    CEDim,
    CellDim,
    EdgeDim,
    KDim,
    V2CDim,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_2nd_order_divergence_of_flux(
    geofac_div: Field[[CEDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    dwdz: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    divergence_wp = neighbor_sum(geofac_div(C2CE) * vn(C2E), axis=C2EDim)
    divergence_wp = divergence_wp + astype(dwdz, wpfloat)
    hexagon_area = neighbor_sum(area(V2C), axis=V2CDim)
    divergence_wp = -divergence_wp * area
    divergence_at_vertex_wp = neighbor_sum(divergence_wp(V2C) / hexagon_area, axis=V2CDim)
    divergence_at_cell_wp = neighbor_sum(divergence_at_vertex_wp(C2V), axis=C2VDim) / 3.0
    return astype(divergence_at_cell_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_2nd_order_divergence_of_flux(
    geofac_div: Field[[CEDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    dwdz: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], wpfloat],
    divergence: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_2nd_order_divergence_of_flux(
        geofac_div,
        vn,
        dwdz,
        area,
        out=divergence,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
