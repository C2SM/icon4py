# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2EC
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_ffsl_backtrajectory(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[wpfloat],
    cell_idx: Field[[dims.ECDim], int32],
    cell_blk: Field[[dims.ECDim], int32],
    edge_verts_1_x: fa.EdgeField[wpfloat],
    edge_verts_2_x: fa.EdgeField[wpfloat],
    edge_verts_1_y: fa.EdgeField[wpfloat],
    edge_verts_2_y: fa.EdgeField[wpfloat],
    pos_on_tplane_e_1_x: fa.EdgeField[wpfloat],
    pos_on_tplane_e_2_x: fa.EdgeField[wpfloat],
    pos_on_tplane_e_1_y: fa.EdgeField[wpfloat],
    pos_on_tplane_e_2_y: fa.EdgeField[wpfloat],
    primal_normal_cell_x: Field[[dims.ECDim], wpfloat],
    primal_normal_cell_y: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_x: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_y: Field[[dims.ECDim], wpfloat],
    lvn_sys_pos: fa.EdgeKField[bool],
    p_dt: wpfloat,
) -> tuple[
    fa.EdgeKField[int32],
    fa.EdgeKField[int32],
    fa.EdgeKField[int32],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    # logical switch for MERGE operations: True for p_vn >= 0
    lvn_pos = where(p_vn >= wpfloat(0.0), True, False)

    # get line and block indices of upwind cell
    p_cell_idx = where(lvn_pos, cell_idx(E2EC[0]), cell_idx(E2EC[1]))
    p_cell_rel_idx_dsl = where(lvn_pos, 0, 1)
    p_cell_blk = where(lvn_pos, cell_blk(E2EC[0]), cell_blk(E2EC[1]))

    # departure points of the departure cell. Point 1 belongs to edge-vertex 1,
    # point 2 belongs to edge_vertex 2.
    #
    # position of vertex 4 (vn > 0) / vertex 2(vn < 0) in normal direction
    depart_pts_1_x = edge_verts_1_x - p_vn * p_dt
    # position of vertex 4 (vn > 0) / vertex 2(vn < 0) in tangential direction
    depart_pts_1_y = edge_verts_1_y - p_vt * p_dt
    # position of vertex 3 in normal direction
    depart_pts_2_x = edge_verts_2_x - p_vn * p_dt
    # position of vertex 3 in tangential direction
    depart_pts_2_y = edge_verts_2_y - p_vt * p_dt

    # determine correct position on tangential plane
    pos_on_tplane_e_x = where(lvn_pos, pos_on_tplane_e_1_x, pos_on_tplane_e_2_x)
    pos_on_tplane_e_y = where(lvn_pos, pos_on_tplane_e_1_y, pos_on_tplane_e_2_y)

    # Calculate position of departure region vertices in a translated
    # coordinate system. The origin is located at the circumcenter
    # of the upwind cell. The distance vectors point from the cell center
    # to the vertices.

    # Take care of correct counterclockwise numbering below

    pos_dreg_vert_c_1_x = edge_verts_1_x - pos_on_tplane_e_x
    pos_dreg_vert_c_1_y = edge_verts_1_y - pos_on_tplane_e_y
    pos_dreg_vert_c_2_x = where(lvn_sys_pos, depart_pts_1_x, edge_verts_2_x) - pos_on_tplane_e_x
    pos_dreg_vert_c_2_y = where(lvn_sys_pos, depart_pts_1_y, edge_verts_2_y) - pos_on_tplane_e_y
    pos_dreg_vert_c_3_x = depart_pts_2_x - pos_on_tplane_e_x
    pos_dreg_vert_c_3_y = depart_pts_2_y - pos_on_tplane_e_y
    pos_dreg_vert_c_4_x = where(lvn_sys_pos, edge_verts_2_x, depart_pts_1_x) - pos_on_tplane_e_x
    pos_dreg_vert_c_4_y = where(lvn_sys_pos, edge_verts_2_y, depart_pts_1_y) - pos_on_tplane_e_y

    # In a last step, these distance vectors are transformed into a rotated
    # geographical coordinate system, which still has its origin at the circumcenter
    # of the upwind cell. Now the coordinate axes point to local East and local
    # North.

    # Determine primal and dual normals of the cell lying in the direction of vn
    pn_cell_1 = where(lvn_pos, primal_normal_cell_x(E2EC[0]), primal_normal_cell_x(E2EC[1]))
    pn_cell_2 = where(lvn_pos, primal_normal_cell_y(E2EC[0]), primal_normal_cell_y(E2EC[1]))
    dn_cell_1 = where(lvn_pos, dual_normal_cell_x(E2EC[0]), dual_normal_cell_x(E2EC[1]))
    dn_cell_2 = where(lvn_pos, dual_normal_cell_y(E2EC[0]), dual_normal_cell_y(E2EC[1]))

    # components in longitudinal direction
    p_coords_dreg_v_1_lon_dsl = pos_dreg_vert_c_1_x * pn_cell_1 + pos_dreg_vert_c_1_y * dn_cell_1
    p_coords_dreg_v_2_lon_dsl = pos_dreg_vert_c_2_x * pn_cell_1 + pos_dreg_vert_c_2_y * dn_cell_1
    p_coords_dreg_v_3_lon_dsl = pos_dreg_vert_c_3_x * pn_cell_1 + pos_dreg_vert_c_3_y * dn_cell_1
    p_coords_dreg_v_4_lon_dsl = pos_dreg_vert_c_4_x * pn_cell_1 + pos_dreg_vert_c_4_y * dn_cell_1

    # components in latitudinal direction
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
def compute_ffsl_backtrajectory(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[wpfloat],
    cell_idx: Field[[dims.ECDim], int32],
    cell_blk: Field[[dims.ECDim], int32],
    edge_verts_1_x: fa.EdgeField[wpfloat],
    edge_verts_2_x: fa.EdgeField[wpfloat],
    edge_verts_1_y: fa.EdgeField[wpfloat],
    edge_verts_2_y: fa.EdgeField[wpfloat],
    pos_on_tplane_e_1_x: fa.EdgeField[wpfloat],
    pos_on_tplane_e_2_x: fa.EdgeField[wpfloat],
    pos_on_tplane_e_1_y: fa.EdgeField[wpfloat],
    pos_on_tplane_e_2_y: fa.EdgeField[wpfloat],
    primal_normal_cell_x: Field[[dims.ECDim], wpfloat],
    primal_normal_cell_y: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_x: Field[[dims.ECDim], wpfloat],
    dual_normal_cell_y: Field[[dims.ECDim], wpfloat],
    lvn_sys_pos: fa.EdgeKField[bool],
    p_dt: wpfloat,
    p_cell_idx: fa.EdgeKField[int32],
    p_cell_rel_idx_dsl: fa.EdgeKField[int32],
    p_cell_blk: fa.EdgeKField[int32],
    p_coords_dreg_v_1_lon_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_2_lon_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_3_lon_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_4_lon_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_1_lat_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_2_lat_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_3_lat_dsl: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_4_lat_dsl: fa.EdgeKField[vpfloat],
):
    _compute_ffsl_backtrajectory(
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
