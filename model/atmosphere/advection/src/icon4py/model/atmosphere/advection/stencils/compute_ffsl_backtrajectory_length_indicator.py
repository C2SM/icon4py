# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2EC
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_ffsl_backtrajectory_length_indicator(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[wpfloat],
    edge_cell_length: Field[[dims.ECDim], wpfloat],
    p_dt: wpfloat,
) -> fa.EdgeKField[int32]:
    lvn_pos = where(p_vn >= wpfloat(0.0), True, False)
    traj_length = sqrt(p_vn * p_vn + p_vt * p_vt) * p_dt
    e2c_length = where(lvn_pos, edge_cell_length(E2EC[0]), edge_cell_length(E2EC[1]))
    opt_famask_dsl = where(
        traj_length > wpfloat(1.25) * broadcast(e2c_length, (dims.EdgeDim, dims.KDim)), 1, 0
    )

    return opt_famask_dsl


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ffsl_backtrajectory_length_indicator(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[wpfloat],
    edge_cell_length: Field[[dims.ECDim], wpfloat],
    p_dt: wpfloat,
    opt_famask_dsl: fa.EdgeKField[int32],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_ffsl_backtrajectory_length_indicator(
        p_vn,
        p_vt,
        edge_cell_length,
        p_dt,
        out=opt_famask_dsl,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
