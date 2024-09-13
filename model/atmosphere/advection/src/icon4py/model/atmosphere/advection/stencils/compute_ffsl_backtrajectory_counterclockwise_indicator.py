# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_ffsl_backtrajectory_counterclockwise_indicator(
    lcounterclock: bool,
    p_vn: fa.EdgeKField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[bool]:
    tangent_orientation = broadcast(tangent_orientation, (dims.EdgeDim, dims.KDim))
    return where(p_vn * tangent_orientation >= wpfloat(0.0), lcounterclock, False)


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ffsl_backtrajectory_counterclockwise_indicator(
    lcounterclock: bool,
    p_vn: fa.EdgeKField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    lvn_sys_pos: fa.EdgeKField[bool],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
