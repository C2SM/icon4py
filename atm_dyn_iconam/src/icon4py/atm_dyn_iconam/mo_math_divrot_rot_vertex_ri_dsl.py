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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import V2E, EdgeDim, KDim, V2EDim, VertexDim


@field_operator
def _mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: Field[[EdgeDim, KDim], float],
    geofac_rot: Field[[VertexDim, V2EDim], float],
) -> Field[[VertexDim, KDim], float]:
    rot_vec = neighbor_sum(vec_e(V2E) * geofac_rot, axis=V2EDim)
    return rot_vec


@program(grid_type=GridType.UNSTRUCTURED)
def mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: Field[[EdgeDim, KDim], float],
    geofac_rot: Field[[VertexDim, V2EDim], float],
    rot_vec: Field[[VertexDim, KDim], float],
):
    _mo_math_divrot_rot_vertex_ri_dsl(vec_e, geofac_rot, out=rot_vec)
