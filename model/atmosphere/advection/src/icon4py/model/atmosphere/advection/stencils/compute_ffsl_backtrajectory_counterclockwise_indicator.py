# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import broadcast, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.settings import backend


@gtx.field_operator
def _compute_ffsl_backtrajectory_counterclockwise_indicator(
    lcounterclock: bool,
    p_vn: fa.EdgeKField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeKField[bool]:
    tangent_orientation = broadcast(tangent_orientation, (dims.EdgeDim, dims.KDim))
    return where(p_vn * tangent_orientation >= 0.0, lcounterclock, False)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_ffsl_backtrajectory_counterclockwise_indicator(
    lcounterclock: bool,
    p_vn: fa.EdgeKField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    lvn_sys_pos: fa.EdgeKField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ffsl_backtrajectory_counterclockwise_indicator(
        lcounterclock,
        p_vn,
        tangent_orientation,
        out=lvn_sys_pos,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
