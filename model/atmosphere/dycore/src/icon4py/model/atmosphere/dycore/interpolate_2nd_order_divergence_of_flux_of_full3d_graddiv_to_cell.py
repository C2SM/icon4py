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

from icon4py.model.common.dimension import C2V, CellDim, VertexDim, C2VDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat
from icon4pytools.icon4pygen.bindings.locations import Vertex


@field_operator
def _interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell(
    z_flxdiv2order_graddiv_vn_vertex: Field[[VertexDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_flxdiv_graddiv_vn = neighbor_sum( z_flxdiv2order_graddiv_vn_vertex(C2V), axis=C2VDim) / 3.0
    return z_flxdiv_graddiv_vn


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell(
    z_flxdiv2order_graddiv_vn_vertex: Field[[VertexDim, KDim], vpfloat],
    z_flxdiv_graddiv_vn: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell(
        z_flxdiv2order_graddiv_vn_vertex,
        out=z_flxdiv_graddiv_vn,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
