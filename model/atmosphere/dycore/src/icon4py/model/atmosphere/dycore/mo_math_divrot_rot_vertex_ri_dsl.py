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

from icon4py.model.common.dimension import V2E, EdgeDim, KDim, V2EDim, VertexDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: Field[[EdgeDim, KDim], wpfloat],
    geofac_rot: Field[[VertexDim, V2EDim], wpfloat],
) -> Field[[VertexDim, KDim], vpfloat]:
    rot_vec_wp = neighbor_sum(vec_e(V2E) * geofac_rot, axis=V2EDim)
    return astype(rot_vec_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: Field[[EdgeDim, KDim], wpfloat],
    geofac_rot: Field[[VertexDim, V2EDim], wpfloat],
    rot_vec: Field[[VertexDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_math_divrot_rot_vertex_ri_dsl(
        vec_e,
        geofac_rot,
        out=rot_vec,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
