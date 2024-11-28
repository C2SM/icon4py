# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory import (
    compute_ffsl_backtrajectory,
)
from icon4py.model.common import dimension as dims


class TestComputeFfslBacktrajectory(helpers.StencilTest):
    PROGRAM = compute_ffsl_backtrajectory
    OUTPUTS = (
        "p_cell_idx",
        "p_cell_rel_idx_dsl",
        "p_cell_blk",
        "p_coords_dreg_v_1_lon_dsl",
        "p_coords_dreg_v_2_lon_dsl",
        "p_coords_dreg_v_3_lon_dsl",
        "p_coords_dreg_v_4_lon_dsl",
        "p_coords_dreg_v_1_lat_dsl",
        "p_coords_dreg_v_2_lat_dsl",
        "p_coords_dreg_v_3_lat_dsl",
        "p_coords_dreg_v_4_lat_dsl",
    )

    @staticmethod
    def reference(
        grid,
        p_vn: np.array,
        p_vt: np.array,
        cell_idx: np.array,
        cell_blk: np.array,
        edge_verts_1_x: np.array,
        edge_verts_2_x: np.array,
        edge_verts_1_y: np.array,
        edge_verts_2_y: np.array,
        pos_on_tplane_e_1_x: np.array,
        pos_on_tplane_e_2_x: np.array,
        pos_on_tplane_e_1_y: np.array,
        pos_on_tplane_e_2_y: np.array,
        primal_normal_cell_x: np.array,
        primal_normal_cell_y: np.array,
        dual_normal_cell_x: np.array,
        dual_normal_cell_y: np.array,
        lvn_sys_pos: np.array,
        p_dt: float,
        **kwargs,
    ) -> dict:
        e2c_shape = grid.connectivities[dims.E2CDim].shape
        cell_idx = helpers.reshape(cell_idx, e2c_shape)
        cell_blk = helpers.reshape(cell_blk, e2c_shape)
        primal_normal_cell_x = helpers.reshape(primal_normal_cell_x, e2c_shape)
        dual_normal_cell_x = helpers.reshape(dual_normal_cell_x, e2c_shape)
        primal_normal_cell_y = helpers.reshape(primal_normal_cell_y, e2c_shape)
        dual_normal_cell_y = helpers.reshape(dual_normal_cell_y, e2c_shape)

        lvn_pos = p_vn >= 0.0
        cell_idx = np.expand_dims(cell_idx, axis=-1)
        cell_blk = np.expand_dims(cell_blk, axis=-1)
        primal_normal_cell_x = np.expand_dims(primal_normal_cell_x, axis=-1)
        dual_normal_cell_x = np.expand_dims(dual_normal_cell_x, axis=-1)
        primal_normal_cell_y = np.expand_dims(primal_normal_cell_y, axis=-1)
        dual_normal_cell_y = np.expand_dims(dual_normal_cell_y, axis=-1)
        edge_verts_1_x = np.expand_dims(edge_verts_1_x, axis=-1)
        edge_verts_1_y = np.expand_dims(edge_verts_1_y, axis=-1)
        edge_verts_2_x = np.expand_dims(edge_verts_2_x, axis=-1)
        edge_verts_2_y = np.expand_dims(edge_verts_2_y, axis=-1)
        pos_on_tplane_e_1_x = np.expand_dims(pos_on_tplane_e_1_x, axis=-1)
        pos_on_tplane_e_1_y = np.expand_dims(pos_on_tplane_e_1_y, axis=-1)
        pos_on_tplane_e_2_x = np.expand_dims(pos_on_tplane_e_2_x, axis=-1)
        pos_on_tplane_e_2_y = np.expand_dims(pos_on_tplane_e_2_y, axis=-1)

        p_cell_idx = np.where(lvn_pos, cell_idx[:, 0], cell_idx[:, 1])
        p_cell_blk = np.where(lvn_pos, cell_blk[:, 0], cell_blk[:, 1])
        p_cell_rel_idx_dsl = np.where(lvn_pos, 0, 1)

        depart_pts_1_x = np.broadcast_to(edge_verts_1_x, p_vn.shape) - p_vn * p_dt
        depart_pts_1_y = np.broadcast_to(edge_verts_1_y, p_vn.shape) - p_vt * p_dt
        depart_pts_2_x = np.broadcast_to(edge_verts_2_x, p_vn.shape) - p_vn * p_dt
        depart_pts_2_y = np.broadcast_to(edge_verts_2_y, p_vn.shape) - p_vt * p_dt

        pos_on_tplane_e_x = np.where(lvn_pos, pos_on_tplane_e_1_x, pos_on_tplane_e_2_x)
        pos_on_tplane_e_y = np.where(lvn_pos, pos_on_tplane_e_1_y, pos_on_tplane_e_2_y)

        pos_dreg_vert_c_1_x = edge_verts_1_x - pos_on_tplane_e_x
        pos_dreg_vert_c_1_y = edge_verts_1_y - pos_on_tplane_e_y
        pos_dreg_vert_c_2_x = (
            np.where(lvn_sys_pos, depart_pts_1_x, edge_verts_2_x) - pos_on_tplane_e_x
        )
        pos_dreg_vert_c_2_y = (
            np.where(lvn_sys_pos, depart_pts_1_y, edge_verts_2_y) - pos_on_tplane_e_y
        )
        pos_dreg_vert_c_3_x = depart_pts_2_x - pos_on_tplane_e_x
        pos_dreg_vert_c_3_y = depart_pts_2_y - pos_on_tplane_e_y
        pos_dreg_vert_c_4_x = (
            np.where(lvn_sys_pos, edge_verts_2_x, depart_pts_1_x) - pos_on_tplane_e_x
        )
        pos_dreg_vert_c_4_y = (
            np.where(lvn_sys_pos, edge_verts_2_y, depart_pts_1_y) - pos_on_tplane_e_y
        )

        pn_cell_1 = np.where(lvn_pos, primal_normal_cell_x[:, 0], primal_normal_cell_x[:, 1])
        pn_cell_2 = np.where(lvn_pos, primal_normal_cell_y[:, 0], primal_normal_cell_y[:, 1])
        dn_cell_1 = np.where(lvn_pos, dual_normal_cell_x[:, 0], dual_normal_cell_x[:, 1])
        dn_cell_2 = np.where(lvn_pos, dual_normal_cell_y[:, 0], dual_normal_cell_y[:, 1])

        p_coords_dreg_v_1_lon_dsl = (
            pos_dreg_vert_c_1_x * pn_cell_1 + pos_dreg_vert_c_1_y * dn_cell_1
        )
        p_coords_dreg_v_2_lon_dsl = (
            pos_dreg_vert_c_2_x * pn_cell_1 + pos_dreg_vert_c_2_y * dn_cell_1
        )
        p_coords_dreg_v_3_lon_dsl = (
            pos_dreg_vert_c_3_x * pn_cell_1 + pos_dreg_vert_c_3_y * dn_cell_1
        )
        p_coords_dreg_v_4_lon_dsl = (
            pos_dreg_vert_c_4_x * pn_cell_1 + pos_dreg_vert_c_4_y * dn_cell_1
        )
        p_coords_dreg_v_1_lat_dsl = (
            pos_dreg_vert_c_1_x * pn_cell_2 + pos_dreg_vert_c_1_y * dn_cell_2
        )
        p_coords_dreg_v_2_lat_dsl = (
            pos_dreg_vert_c_2_x * pn_cell_2 + pos_dreg_vert_c_2_y * dn_cell_2
        )
        p_coords_dreg_v_3_lat_dsl = (
            pos_dreg_vert_c_3_x * pn_cell_2 + pos_dreg_vert_c_3_y * dn_cell_2
        )
        p_coords_dreg_v_4_lat_dsl = (
            pos_dreg_vert_c_4_x * pn_cell_2 + pos_dreg_vert_c_4_y * dn_cell_2
        )
        return {
            "p_cell_idx": p_cell_idx,
            "p_cell_rel_idx_dsl": p_cell_rel_idx_dsl,
            "p_cell_blk": p_cell_blk,
            "p_coords_dreg_v_1_lon_dsl": p_coords_dreg_v_1_lon_dsl,
            "p_coords_dreg_v_2_lon_dsl": p_coords_dreg_v_2_lon_dsl,
            "p_coords_dreg_v_3_lon_dsl": p_coords_dreg_v_3_lon_dsl,
            "p_coords_dreg_v_4_lon_dsl": p_coords_dreg_v_4_lon_dsl,
            "p_coords_dreg_v_1_lat_dsl": p_coords_dreg_v_1_lat_dsl,
            "p_coords_dreg_v_2_lat_dsl": p_coords_dreg_v_2_lat_dsl,
            "p_coords_dreg_v_3_lat_dsl": p_coords_dreg_v_3_lat_dsl,
            "p_coords_dreg_v_4_lat_dsl": p_coords_dreg_v_4_lat_dsl,
        }

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        cell_idx = grid.connectivities[dims.E2CDim]
        cell_idx_new = helpers.numpy_to_1D_sparse_field(cell_idx, dims.ECDim)
        cell_blk = helpers.constant_field(grid, 1, dims.EdgeDim, dims.E2CDim, dtype=gtx.int32)
        cell_blk_new = helpers.as_1D_sparse_field(cell_blk, dims.ECDim)

        edge_verts_1_x = helpers.random_field(grid, dims.EdgeDim)
        edge_verts_2_x = helpers.random_field(grid, dims.EdgeDim)
        edge_verts_1_y = helpers.random_field(grid, dims.EdgeDim)
        edge_verts_2_y = helpers.random_field(grid, dims.EdgeDim)
        pos_on_tplane_e_1_x = helpers.random_field(grid, dims.EdgeDim)
        pos_on_tplane_e_2_x = helpers.random_field(grid, dims.EdgeDim)
        pos_on_tplane_e_1_y = helpers.random_field(grid, dims.EdgeDim)
        pos_on_tplane_e_2_y = helpers.random_field(grid, dims.EdgeDim)
        primal_normal_cell_x = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_x_new = helpers.as_1D_sparse_field(primal_normal_cell_x, dims.ECDim)
        dual_normal_cell_x = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_x_new = helpers.as_1D_sparse_field(dual_normal_cell_x, dims.ECDim)
        primal_normal_cell_y = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_y_new = helpers.as_1D_sparse_field(primal_normal_cell_y, dims.ECDim)
        dual_normal_cell_y = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_y_new = helpers.as_1D_sparse_field(dual_normal_cell_y, dims.ECDim)
        lvn_sys_pos = helpers.constant_field(grid, True, dims.EdgeDim, dims.KDim, dtype=bool)
        p_dt = 2.0
        p_cell_idx = helpers.constant_field(grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_cell_rel_idx_dsl = helpers.constant_field(
            grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32
        )
        p_cell_blk = helpers.constant_field(grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_coords_dreg_v_1_lon_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_2_lon_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_3_lon_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_4_lon_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_1_lat_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_2_lat_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_3_lat_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_4_lat_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            cell_idx=cell_idx_new,
            cell_blk=cell_blk_new,
            edge_verts_1_x=edge_verts_1_x,
            edge_verts_2_x=edge_verts_2_x,
            edge_verts_1_y=edge_verts_1_y,
            edge_verts_2_y=edge_verts_2_y,
            pos_on_tplane_e_1_x=pos_on_tplane_e_1_x,
            pos_on_tplane_e_2_x=pos_on_tplane_e_2_x,
            pos_on_tplane_e_1_y=pos_on_tplane_e_1_y,
            pos_on_tplane_e_2_y=pos_on_tplane_e_2_y,
            primal_normal_cell_x=primal_normal_cell_x_new,
            primal_normal_cell_y=primal_normal_cell_y_new,
            dual_normal_cell_x=dual_normal_cell_x_new,
            dual_normal_cell_y=dual_normal_cell_y_new,
            lvn_sys_pos=lvn_sys_pos,
            p_cell_idx=p_cell_idx,
            p_cell_rel_idx_dsl=p_cell_rel_idx_dsl,
            p_cell_blk=p_cell_blk,
            p_coords_dreg_v_1_lon_dsl=p_coords_dreg_v_1_lon_dsl,
            p_coords_dreg_v_2_lon_dsl=p_coords_dreg_v_2_lon_dsl,
            p_coords_dreg_v_3_lon_dsl=p_coords_dreg_v_3_lon_dsl,
            p_coords_dreg_v_4_lon_dsl=p_coords_dreg_v_4_lon_dsl,
            p_coords_dreg_v_1_lat_dsl=p_coords_dreg_v_1_lat_dsl,
            p_coords_dreg_v_2_lat_dsl=p_coords_dreg_v_2_lat_dsl,
            p_coords_dreg_v_3_lat_dsl=p_coords_dreg_v_3_lat_dsl,
            p_coords_dreg_v_4_lat_dsl=p_coords_dreg_v_4_lat_dsl,
            p_dt=p_dt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
