# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_ffsl_backtrajectory_counterclockwise_indicator(
    lcounterclock: bool,
    p_vn: fa.EdgeKField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[bool]:
    tangent_orientation = broadcast(tangent_orientation, (EdgeDim, KDim))
    return where(p_vn * tangent_orientation >= wpfloat(0.0), lcounterclock, False)


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ffsl_backtrajectory_counterclockwise_indicator(
    lcounterclock: bool,
    p_vn: fa.EdgeKField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    lvn_sys_pos: fa.EdgeKField[bool],
):
    _compute_ffsl_backtrajectory_counterclockwise_indicator(
        lcounterclock, p_vn, tangent_orientation, out=lvn_sys_pos
    )
