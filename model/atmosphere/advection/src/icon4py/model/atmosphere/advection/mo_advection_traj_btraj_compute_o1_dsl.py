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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, where

from icon4py.model.common.dimension import E2EC, ECDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _mo_advection_traj_btraj_compute_o1_dsl(
    p_vn: Field[[EdgeDim, KDim], wpfloat],
    p_vt: Field[[EdgeDim, KDim], wpfloat],
    cell_idx: Field[[ECDim], int32],
    cell_blk: Field[[ECDim], int32],
    pos_on_tplane_e_1: Field[[ECDim], wpfloat],
    pos_on_tplane_e_2: Field[[ECDim], wpfloat],
    primal_normal_cell_1: Field[[ECDim], wpfloat],
    dual_normal_cell_1: Field[[ECDim], wpfloat],
    primal_normal_cell_2: Field[[ECDim], wpfloat],
    dual_normal_cell_2: Field[[ECDim], wpfloat],
    p_dthalf: wpfloat,
) -> tuple[
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], vpfloat],
    Field[[EdgeDim, KDim], vpfloat],
]:
    lvn_pos = where(p_vn > 0.0, True, False)

    p_cell_idx = where(lvn_pos, cell_idx(E2EC[0]), cell_idx(E2EC[1]))
    p_cell_rel_idx_dsl = where(lvn_pos, 0, 1)
    p_cell_blk = where(lvn_pos, cell_blk(E2EC[0]), cell_blk(E2EC[1]))

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

    return (
        p_cell_idx,
        p_cell_rel_idx_dsl,
        p_cell_blk,
        astype(p_distv_bary_1, vpfloat),
        astype(p_distv_bary_2, vpfloat),
    )


@program(grid_type=GridType.UNSTRUCTURED)
def mo_advection_traj_btraj_compute_o1_dsl(
    p_vn: Field[[EdgeDim, KDim], wpfloat],
    p_vt: Field[[EdgeDim, KDim], wpfloat],
    cell_idx: Field[[ECDim], int32],
    cell_blk: Field[[ECDim], int32],
    pos_on_tplane_e_1: Field[[ECDim], wpfloat],
    pos_on_tplane_e_2: Field[[ECDim], wpfloat],
    primal_normal_cell_1: Field[[ECDim], wpfloat],
    dual_normal_cell_1: Field[[ECDim], wpfloat],
    primal_normal_cell_2: Field[[ECDim], wpfloat],
    dual_normal_cell_2: Field[[ECDim], wpfloat],
    p_cell_idx: Field[[EdgeDim, KDim], int32],
    p_cell_rel_idx_dsl: Field[[EdgeDim, KDim], int32],
    p_cell_blk: Field[[EdgeDim, KDim], int32],
    p_distv_bary_1: Field[[EdgeDim, KDim], vpfloat],
    p_distv_bary_2: Field[[EdgeDim, KDim], vpfloat],
    p_dthalf: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_advection_traj_btraj_compute_o1_dsl(
        p_vn,
        p_vt,
        cell_idx,
        cell_blk,
        pos_on_tplane_e_1,
        pos_on_tplane_e_2,
        primal_normal_cell_1,
        dual_normal_cell_1,
        primal_normal_cell_2,
        dual_normal_cell_2,
        p_dthalf,
        out=(
            p_cell_idx,
            p_cell_rel_idx_dsl,
            p_cell_blk,
            p_distv_bary_1,
            p_distv_bary_2,
        ),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
