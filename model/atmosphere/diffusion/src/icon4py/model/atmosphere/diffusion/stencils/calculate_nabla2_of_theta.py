# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_nabla2_of_theta(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
) -> fa.CellKField[vpfloat]:
    z_temp_wp = neighbor_sum(z_nabla2_e(C2E) * geofac_div(C2CE), axis=C2EDim)
    return astype(z_temp_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_of_theta(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_temp: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_nabla2_of_theta(
        z_nabla2_e,
        geofac_div,
        out=z_temp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
