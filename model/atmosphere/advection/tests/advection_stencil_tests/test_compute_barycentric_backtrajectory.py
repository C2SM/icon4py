# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_barycentric_backtrajectory import (
    compute_barycentric_backtrajectory,
)
from icon4py.model.common import dimension as dims
import numpy as xp


class TestComputeBarycentricBacktrajectory(helpers.StencilTest):
    PROGRAM = compute_barycentric_backtrajectory
    OUTPUTS = ("p_cell_idx", "p_cell_blk", "p_distv_bary_1", "p_distv_bary_2")

    @staticmethod
    def reference(
        grid,
        p_vn: xp.array,
        p_vt: xp.array,
        cell_idx: xp.array,
        cell_blk: xp.array,
        pos_on_tplane_e_1: xp.array,
        pos_on_tplane_e_2: xp.array,
        primal_normal_cell_1: xp.array,
        dual_normal_cell_1: xp.array,
        primal_normal_cell_2: xp.array,
        dual_normal_cell_2: xp.array,
        p_dthalf: float,
        **kwargs,
    ) -> dict:
        e2c = grid.connectivities[dims.E2CDim]
        cell_idx = cell_idx.reshape(e2c.shape)
        cell_blk = cell_blk.reshape(e2c.shape)
        pos_on_tplane_e_1 = pos_on_tplane_e_1.reshape(e2c.shape)
        pos_on_tplane_e_2 = pos_on_tplane_e_2.reshape(e2c.shape)
        primal_normal_cell_1 = primal_normal_cell_1.reshape(e2c.shape)
        primal_normal_cell_2 = primal_normal_cell_2.reshape(e2c.shape)
        dual_normal_cell_1 = dual_normal_cell_1.reshape(e2c.shape)
        dual_normal_cell_2 = dual_normal_cell_2.reshape(e2c.shape)

        lvn_pos = p_vn >= 0.0
        cell_idx = xp.expand_dims(xp.asarray(cell_idx), axis=-1)
        cell_blk = xp.expand_dims(xp.asarray(cell_blk), axis=-1)
        pos_on_tplane_e_1 = xp.expand_dims(xp.asarray(pos_on_tplane_e_1), axis=-1)
        pos_on_tplane_e_2 = xp.expand_dims(xp.asarray(pos_on_tplane_e_2), axis=-1)
        primal_normal_cell_1 = xp.expand_dims(xp.asarray(primal_normal_cell_1), axis=-1)
        dual_normal_cell_1 = xp.expand_dims(xp.asarray(dual_normal_cell_1), axis=-1)
        primal_normal_cell_2 = xp.expand_dims(xp.asarray(primal_normal_cell_2), axis=-1)
        dual_normal_cell_2 = xp.expand_dims(xp.asarray(dual_normal_cell_2), axis=-1)

        p_cell_idx = xp.where(lvn_pos, cell_idx[:, 0], cell_idx[:, 1])
        p_cell_rel_idx_dsl = xp.where(lvn_pos, 0, 1)
        p_cell_blk = xp.where(lvn_pos, cell_blk[:, 0], cell_blk[:, 1])

        z_ntdistv_bary_1 = -(
            p_vn * p_dthalf + xp.where(lvn_pos, pos_on_tplane_e_1[:, 0], pos_on_tplane_e_1[:, 1])
        )
        z_ntdistv_bary_2 = -(
            p_vt * p_dthalf + xp.where(lvn_pos, pos_on_tplane_e_2[:, 0], pos_on_tplane_e_2[:, 1])
        )

        p_distv_bary_1 = xp.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 1],
        )

        p_distv_bary_2 = xp.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_2[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_2[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_2[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_2[:, 1],
        )

        return dict(
            p_cell_idx=p_cell_idx,
            p_cell_rel_idx_dsl=p_cell_rel_idx_dsl,
            p_cell_blk=p_cell_blk,
            p_distv_bary_1=p_distv_bary_1,
            p_distv_bary_2=p_distv_bary_2,
        )

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        cell_idx = xp.asarray(grid.connectivities[dims.E2CDim], dtype=gtx.int32)
        cell_idx_new = helpers.numpy_to_1D_sparse_field(cell_idx, dims.ECDim)
        cell_blk = helpers.constant_field(grid, 1, dims.EdgeDim, dims.E2CDim, dtype=gtx.int32)
        cell_blk_new = helpers.as_1D_sparse_field(cell_blk, dims.ECDim)
        pos_on_tplane_e_1 = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        pos_on_tplane_e_1_new = helpers.as_1D_sparse_field(pos_on_tplane_e_1, dims.ECDim)
        pos_on_tplane_e_2 = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        pos_on_tplane_e_2_new = helpers.as_1D_sparse_field(pos_on_tplane_e_2, dims.ECDim)
        primal_normal_cell_1 = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_1_new = helpers.as_1D_sparse_field(primal_normal_cell_1, dims.ECDim)
        dual_normal_cell_1 = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_1_new = helpers.as_1D_sparse_field(dual_normal_cell_1, dims.ECDim)
        primal_normal_cell_2 = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_2_new = helpers.as_1D_sparse_field(primal_normal_cell_2, dims.ECDim)
        dual_normal_cell_2 = helpers.random_field(grid, dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_2_new = helpers.as_1D_sparse_field(dual_normal_cell_2, dims.ECDim)
        p_cell_idx = helpers.constant_field(grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_cell_rel_idx_dsl = helpers.constant_field(
            grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32
        )
        p_cell_blk = helpers.constant_field(grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_distv_bary_1 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_distv_bary_2 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_dthalf = 2.0

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            cell_idx=cell_idx_new,
            cell_blk=cell_blk_new,
            pos_on_tplane_e_1=pos_on_tplane_e_1_new,
            pos_on_tplane_e_2=pos_on_tplane_e_2_new,
            primal_normal_cell_1=primal_normal_cell_1_new,
            dual_normal_cell_1=dual_normal_cell_1_new,
            primal_normal_cell_2=primal_normal_cell_2_new,
            dual_normal_cell_2=dual_normal_cell_2_new,
            p_cell_idx=p_cell_idx,
            p_cell_rel_idx_dsl=p_cell_rel_idx_dsl,
            p_cell_blk=p_cell_blk,
            p_distv_bary_1=p_distv_bary_1,
            p_distv_bary_2=p_distv_bary_2,
            p_dthalf=p_dthalf,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
