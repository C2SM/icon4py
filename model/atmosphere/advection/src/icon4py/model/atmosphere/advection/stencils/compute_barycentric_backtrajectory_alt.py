# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2EC
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_barycentric_backtrajectory_alt(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[wpfloat],
    pos_on_tplane_e_1: Field[[dims.ECDim], wpfloat],
    pos_on_tplane_e_2: Field[[dims.ECDim], wpfloat],
    primal_normal_cell_1: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_1: Field[[dims.ECDim], wpfloat],
    primal_normal_cell_2: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_2: Field[[dims.ECDim], wpfloat],
    p_dthalf: wpfloat,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    lvn_pos = where(p_vn > wpfloat(0.0), True, False)

    z_ntdistv_bary_1 = -(
        p_vn * p_dthalf + where(lvn_pos, pos_on_tplane_e_1(E2EC[0]), pos_on_tplane_e_1(E2EC[1]))
    )

    z_ntdistv_bary_2 = -(
        p_vt * p_dthalf + where(lvn_pos, pos_on_tplane_e_2(E2EC[0]), pos_on_tplane_e_2(E2EC[1]))
    )

    p_distv_bary_1 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_1(E2EC[0])
        + z_ntdistv_bary_2 * dual_normal_cell_1(E2EC[0]),
        z_ntdistv_bary_1 * primal_normal_cell_1(E2EC[1])
        + z_ntdistv_bary_2 * dual_normal_cell_1(E2EC[1]),
    )

    p_distv_bary_2 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_2(E2EC[0])
        + z_ntdistv_bary_2 * dual_normal_cell_2(E2EC[0]),
        z_ntdistv_bary_1 * primal_normal_cell_2(E2EC[1])
        + z_ntdistv_bary_2 * dual_normal_cell_2(E2EC[1]),
    )

    return (astype(p_distv_bary_1, vpfloat), astype(p_distv_bary_2, vpfloat))


@program(grid_type=GridType.UNSTRUCTURED)
def compute_barycentric_backtrajectory_alt(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[wpfloat],
    pos_on_tplane_e_1: Field[[dims.ECDim], wpfloat],
    pos_on_tplane_e_2: Field[[dims.ECDim], wpfloat],
    primal_normal_cell_1: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_1: Field[[dims.ECDim], wpfloat],
    primal_normal_cell_2: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_2: Field[[dims.ECDim], wpfloat],
    p_distv_bary_1: fa.EdgeKField[vpfloat],
    p_distv_bary_2: fa.EdgeKField[vpfloat],
    p_dthalf: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_barycentric_backtrajectory_alt(
        p_vn,
        p_vt,
        pos_on_tplane_e_1,
        pos_on_tplane_e_2,
        primal_normal_cell_1,
        dual_normal_cell_1,
        primal_normal_cell_2,
        dual_normal_cell_2,
        p_dthalf,
        out=(p_distv_bary_1, p_distv_bary_2),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
