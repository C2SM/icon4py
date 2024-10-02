# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff


@field_operator
def _face_val_ppm_stencil_02a(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
) -> fa.CellKField[float]:
    p_face = p_cc * (1.0 - (p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1]))) + (
        p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now)
    ) * ((p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1])) * p_cc + p_cc(Koff[-1]))

    return p_face


@program(grid_type=GridType.UNSTRUCTURED)
def face_val_ppm_stencil_02a(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
    p_face: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _face_val_ppm_stencil_02a(
        p_cc,
        p_cellhgt_mc_now,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
