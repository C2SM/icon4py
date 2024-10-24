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
from icon4py.model.atmosphere.advection.stencils.compute_barycentric_backtrajectory_alt import (
    compute_barycentric_backtrajectory_alt,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp


class TestComputeBarycentricBacktrajectoryAlt(helpers.StencilTest):
    PROGRAM = compute_barycentric_backtrajectory_alt
    OUTPUTS = ("p_distv_bary_1", "p_distv_bary_2")

    @staticmethod
    def reference(
        grid,
        p_vn: xp.array,
        p_vt: xp.array,
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
        pos_on_tplane_e_1 = pos_on_tplane_e_1.reshape(e2c.shape)
        pos_on_tplane_e_2 = pos_on_tplane_e_2.reshape(e2c.shape)
        primal_normal_cell_1 = primal_normal_cell_1.reshape(e2c.shape)
        primal_normal_cell_2 = primal_normal_cell_2.reshape(e2c.shape)
        dual_normal_cell_1 = dual_normal_cell_1.reshape(e2c.shape)
        dual_normal_cell_2 = dual_normal_cell_2.reshape(e2c.shape)

        lvn_pos = p_vn >= 0.0
        pos_on_tplane_e_1 = xp.expand_dims(pos_on_tplane_e_1, axis=-1)
        pos_on_tplane_e_2 = xp.expand_dims(pos_on_tplane_e_2, axis=-1)
        primal_normal_cell_1 = xp.expand_dims(primal_normal_cell_1, axis=-1)
        dual_normal_cell_1 = xp.expand_dims(dual_normal_cell_1, axis=-1)
        primal_normal_cell_2 = xp.expand_dims(primal_normal_cell_2, axis=-1)
        dual_normal_cell_2 = xp.expand_dims(dual_normal_cell_2, axis=-1)

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
            p_distv_bary_1=p_distv_bary_1,
            p_distv_bary_2=p_distv_bary_2,
        )

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
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
        p_distv_bary_1 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_distv_bary_2 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_dthalf = 2.0

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1_new,
            pos_on_tplane_e_2=pos_on_tplane_e_2_new,
            primal_normal_cell_1=primal_normal_cell_1_new,
            dual_normal_cell_1=dual_normal_cell_1_new,
            primal_normal_cell_2=primal_normal_cell_2_new,
            dual_normal_cell_2=dual_normal_cell_2_new,
            p_distv_bary_1=p_distv_bary_1,
            p_distv_bary_2=p_distv_bary_2,
            p_dthalf=p_dthalf,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
