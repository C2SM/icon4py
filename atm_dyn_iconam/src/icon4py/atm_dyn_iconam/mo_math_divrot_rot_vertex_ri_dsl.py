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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import (
    V2E,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)


@field_operator
def _mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: Field[[EdgeDim, KDim], float],
    geofac_rot: Field[[VertexDim, V2EDim], float],
) -> Field[[VertexDim, KDim], float]:
    rot_vec = neighbor_sum(geofac_rot * vec_e(V2E), axis=V2EDim)
    return rot_vec


@program
def mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: Field[[EdgeDim, KDim], float],
    geofac_rot: Field[[VertexDim, V2EDim], float],
    rot_vec: Field[[VertexDim, KDim], float],
):
    _mo_math_divrot_rot_vertex_ri_dsl(vec_e, geofac_rot, out=rot_vec)
