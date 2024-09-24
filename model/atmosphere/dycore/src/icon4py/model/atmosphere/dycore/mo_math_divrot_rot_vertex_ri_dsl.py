# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import V2E, V2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_rot: Field[[dims.VertexDim, V2EDim], wpfloat],
) -> fa.VertexKField[vpfloat]:
    rot_vec_wp = neighbor_sum(vec_e(V2E) * geofac_rot, axis=V2EDim)
    return astype(rot_vec_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_rot: Field[[dims.VertexDim, V2EDim], wpfloat],
    rot_vec: fa.VertexKField[vpfloat],
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
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
