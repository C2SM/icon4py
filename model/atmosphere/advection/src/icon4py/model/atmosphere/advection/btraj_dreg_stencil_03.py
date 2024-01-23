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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.common.dimension import E2EC, ECDim, EdgeDim, KDim


@field_operator
def _btraj_dreg_stencil_03(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    cell_idx: Field[[ECDim], int32],
    cell_blk: Field[[ECDim], int32],
    edge_verts_1_x: Field[[EdgeDim], float],
    edge_verts_2_x: Field[[EdgeDim], float],
    edge_verts_1_y: Field[[EdgeDim], float],
    edge_verts_2_y: Field[[EdgeDim], float],
    pos_on_tplane_e_1_x: Field[[EdgeDim], float],
    pos_on_tplane_e_2_x: Field[[EdgeDim], float],
    pos_on_tplane_e_1_y: Field[[EdgeDim], float],
    pos_on_tplane_e_2_y: Field[[EdgeDim], float],
    primal_normal_cell_x: Field[[ECDim], float],
    primal_normal_cell_y: Field[[ECDim], float],
    dual_normal_cell_x: Field[[ECDim], float],
    dual_normal_cell_y: Field[[ECDim], float],
    lvn_sys_pos: Field[[EdgeDim, KDim], bool],
    p_dt: float,
) -> tuple[
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    lvn_pos = where(p_vn >= 0.0, True, False)

    p_cell_idx = where(lvn_pos, cell_idx(E2EC[0]), cell_idx(E2EC[1]))
    p_cell_rel_idx_dsl = where(lvn_pos, int32(0), int32(1))
    p_cell_blk = where(lvn_pos, cell_blk(E2EC[0]), cell_blk(E2EC[1]))

    depart_pts_1_x = edge_verts_1_x - p_vn * p_dt
    depart_pts_1_y = edge_verts_1_y - p_vt * p_dt
    depart_pts_2_x = edge_verts_2_x - p_vn * p_dt
    depart_pts_2_y = edge_verts_2_y - p_vt * p_dt

    pos_on_tplane_e_x = where(lvn_pos, pos_on_tplane_e_1_x, pos_on_tplane_e_2_x)
    pos_on_tplane_e_y = where(lvn_pos, pos_on_tplane_e_1_y, pos_on_tplane_e_2_y)

    pos_dreg_vert_c_1_x = edge_verts_1_x - pos_on_tplane_e_x
    pos_dreg_vert_c_1_y = edge_verts_1_y - pos_on_tplane_e_y
    pos_dreg_vert_c_2_x = where(lvn_sys_pos, depart_pts_1_x, edge_verts_2_x) - pos_on_tplane_e_x
    pos_dreg_vert_c_2_y = where(lvn_sys_pos, depart_pts_1_y, edge_verts_2_y) - pos_on_tplane_e_y
    pos_dreg_vert_c_3_x = depart_pts_2_x - pos_on_tplane_e_x
    pos_dreg_vert_c_3_y = depart_pts_2_y - pos_on_tplane_e_y
    pos_dreg_vert_c_4_x = where(lvn_sys_pos, edge_verts_2_x, depart_pts_1_x) - pos_on_tplane_e_x
    pos_dreg_vert_c_4_y = where(lvn_sys_pos, edge_verts_2_y, depart_pts_1_y) - pos_on_tplane_e_y

    pn_cell_1 = where(lvn_pos, primal_normal_cell_x(E2EC[0]), primal_normal_cell_x(E2EC[1]))
    pn_cell_2 = where(lvn_pos, primal_normal_cell_y(E2EC[0]), primal_normal_cell_y(E2EC[1]))
    dn_cell_1 = where(lvn_pos, dual_normal_cell_x(E2EC[0]), dual_normal_cell_x(E2EC[1]))
    dn_cell_2 = where(lvn_pos, dual_normal_cell_y(E2EC[0]), dual_normal_cell_y(E2EC[1]))

    p_coords_dreg_v_1_lon_dsl = pos_dreg_vert_c_1_x * pn_cell_1 + pos_dreg_vert_c_1_y * dn_cell_1
    p_coords_dreg_v_2_lon_dsl = pos_dreg_vert_c_2_x * pn_cell_1 + pos_dreg_vert_c_2_y * dn_cell_1
    p_coords_dreg_v_3_lon_dsl = pos_dreg_vert_c_3_x * pn_cell_1 + pos_dreg_vert_c_3_y * dn_cell_1
    p_coords_dreg_v_4_lon_dsl = pos_dreg_vert_c_4_x * pn_cell_1 + pos_dreg_vert_c_4_y * dn_cell_1
    p_coords_dreg_v_1_lat_dsl = pos_dreg_vert_c_1_x * pn_cell_2 + pos_dreg_vert_c_1_y * dn_cell_2
    p_coords_dreg_v_2_lat_dsl = pos_dreg_vert_c_2_x * pn_cell_2 + pos_dreg_vert_c_2_y * dn_cell_2
    p_coords_dreg_v_3_lat_dsl = pos_dreg_vert_c_3_x * pn_cell_2 + pos_dreg_vert_c_3_y * dn_cell_2
    p_coords_dreg_v_4_lat_dsl = pos_dreg_vert_c_4_x * pn_cell_2 + pos_dreg_vert_c_4_y * dn_cell_2

    return (
        p_cell_idx,
        p_cell_rel_idx_dsl,
        p_cell_blk,
        p_coords_dreg_v_1_lon_dsl,
        p_coords_dreg_v_2_lon_dsl,
        p_coords_dreg_v_3_lon_dsl,
        p_coords_dreg_v_4_lon_dsl,
        p_coords_dreg_v_1_lat_dsl,
        p_coords_dreg_v_2_lat_dsl,
        p_coords_dreg_v_3_lat_dsl,
        p_coords_dreg_v_4_lat_dsl,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def btraj_dreg_stencil_03(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    cell_idx: Field[[ECDim], int32],
    cell_blk: Field[[ECDim], int32],
    edge_verts_1_x: Field[[EdgeDim], float],
    edge_verts_2_x: Field[[EdgeDim], float],
    edge_verts_1_y: Field[[EdgeDim], float],
    edge_verts_2_y: Field[[EdgeDim], float],
    pos_on_tplane_e_1_x: Field[[EdgeDim], float],
    pos_on_tplane_e_2_x: Field[[EdgeDim], float],
    pos_on_tplane_e_1_y: Field[[EdgeDim], float],
    pos_on_tplane_e_2_y: Field[[EdgeDim], float],
    primal_normal_cell_x: Field[[ECDim], float],
    primal_normal_cell_y: Field[[ECDim], float],
    dual_normal_cell_x: Field[[ECDim], float],
    dual_normal_cell_y: Field[[ECDim], float],
    lvn_sys_pos: Field[[EdgeDim, KDim], bool],
    p_dt: float,
    p_cell_idx: Field[[EdgeDim, KDim], int32],
    p_cell_rel_idx_dsl: Field[[EdgeDim, KDim], int32],
    p_cell_blk: Field[[EdgeDim, KDim], int32],
    p_coords_dreg_v_1_lon_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_2_lon_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_3_lon_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_4_lon_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_1_lat_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_2_lat_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_3_lat_dsl: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_4_lat_dsl: Field[[EdgeDim, KDim], float],
):
    _btraj_dreg_stencil_03(
        p_vn,
        p_vt,
        cell_idx,
        cell_blk,
        edge_verts_1_x,
        edge_verts_2_x,
        edge_verts_1_y,
        edge_verts_2_y,
        pos_on_tplane_e_1_x,
        pos_on_tplane_e_2_x,
        pos_on_tplane_e_1_y,
        pos_on_tplane_e_2_y,
        primal_normal_cell_x,
        primal_normal_cell_y,
        dual_normal_cell_x,
        dual_normal_cell_y,
        lvn_sys_pos,
        p_dt,
        out=(
            p_cell_idx,
            p_cell_rel_idx_dsl,
            p_cell_blk,
            p_coords_dreg_v_1_lon_dsl,
            p_coords_dreg_v_2_lon_dsl,
            p_coords_dreg_v_3_lon_dsl,
            p_coords_dreg_v_4_lon_dsl,
            p_coords_dreg_v_1_lat_dsl,
            p_coords_dreg_v_2_lat_dsl,
            p_coords_dreg_v_3_lat_dsl,
            p_coords_dreg_v_4_lat_dsl,
        ),
    )
