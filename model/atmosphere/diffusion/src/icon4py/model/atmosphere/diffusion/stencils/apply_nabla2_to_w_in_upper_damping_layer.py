# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CellKField[wpfloat],
    diff_multfac_n2w: fa.KField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
) -> fa.CellKField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)
    cell_area_tmp = broadcast(cell_area, (dims.CellDim, dims.KDim))

    w_wp = w + diff_multfac_n2w * cell_area_tmp * z_nabla2_c_wp
    return w_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CellKField[wpfloat],
    diff_multfac_n2w: fa.KField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_nabla2_to_w_in_upper_damping_layer(
        w,
        diff_multfac_n2w,
        cell_area,
        z_nabla2_c,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
