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

import icon4py.model.common.utils.data_allocation as data_alloc
import icon4py.model.testing.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_barycentric_backtrajectory import (
    compute_barycentric_backtrajectory,
)
from icon4py.model.common import dimension as dims


class TestComputeBarycentricBacktrajectory(helpers.StencilTest):
    PROGRAM = compute_barycentric_backtrajectory
    OUTPUTS = ("p_cell_idx", "p_cell_blk", "p_distv_bary_1", "p_distv_bary_2")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        p_vn: np.ndarray,
        p_vt: np.ndarray,
        cell_idx: np.ndarray,
        cell_blk: np.ndarray,
        pos_on_tplane_e_1: np.ndarray,
        pos_on_tplane_e_2: np.ndarray,
        primal_normal_cell_1: np.ndarray,
        dual_normal_cell_1: np.ndarray,
        primal_normal_cell_2: np.ndarray,
        dual_normal_cell_2: np.ndarray,
        p_dthalf: float,
        **kwargs,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]
        cell_idx = cell_idx.reshape(e2c.shape)
        cell_blk = cell_blk.reshape(e2c.shape)
        pos_on_tplane_e_1 = pos_on_tplane_e_1.reshape(e2c.shape)
        pos_on_tplane_e_2 = pos_on_tplane_e_2.reshape(e2c.shape)
        primal_normal_cell_1 = primal_normal_cell_1.reshape(e2c.shape)
        primal_normal_cell_2 = primal_normal_cell_2.reshape(e2c.shape)
        dual_normal_cell_1 = dual_normal_cell_1.reshape(e2c.shape)
        dual_normal_cell_2 = dual_normal_cell_2.reshape(e2c.shape)

        lvn_pos = p_vn >= 0.0
        cell_idx = np.expand_dims(cell_idx, axis=-1)
        cell_blk = np.expand_dims(cell_blk, axis=-1)
        pos_on_tplane_e_1 = np.expand_dims(pos_on_tplane_e_1, axis=-1)
        pos_on_tplane_e_2 = np.expand_dims(pos_on_tplane_e_2, axis=-1)
        primal_normal_cell_1 = np.expand_dims(primal_normal_cell_1, axis=-1)
        dual_normal_cell_1 = np.expand_dims(dual_normal_cell_1, axis=-1)
        primal_normal_cell_2 = np.expand_dims(primal_normal_cell_2, axis=-1)
        dual_normal_cell_2 = np.expand_dims(dual_normal_cell_2, axis=-1)

        p_cell_idx = np.where(lvn_pos, cell_idx[:, 0], cell_idx[:, 1])
        p_cell_rel_idx_dsl = np.where(lvn_pos, 0, 1)
        p_cell_blk = np.where(lvn_pos, cell_blk[:, 0], cell_blk[:, 1])

        z_ntdistv_bary_1 = -(
            p_vn * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_1[:, 0], pos_on_tplane_e_1[:, 1])
        )
        z_ntdistv_bary_2 = -(
            p_vt * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_2[:, 0], pos_on_tplane_e_2[:, 1])
        )

        p_distv_bary_1 = np.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 1],
        )

        p_distv_bary_2 = np.where(
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
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        cell_idx = data_alloc.flatten_first_two_dims(
            dims.ECDim, field=(grid.connectivities[dims.E2CDim])
        )
        cell_blk = data_alloc.flatten_first_two_dims(
            dims.ECDim,
            field=(data_alloc.constant_field(grid, 1, dims.EdgeDim, dims.E2CDim, dtype=gtx.int32)),
        )
        pos_on_tplane_e_1 = data_alloc.random_field(grid, dims.ECDim)
        pos_on_tplane_e_2 = data_alloc.random_field(grid, dims.ECDim)
        primal_normal_cell_1 = data_alloc.random_field(grid, dims.ECDim)

        dual_normal_cell_1 = data_alloc.random_field(grid, dims.ECDim)
        primal_normal_cell_2 = data_alloc.random_field(grid, dims.ECDim)

        dual_normal_cell_2 = data_alloc.random_field(grid, dims.ECDim)

        p_cell_idx = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_cell_rel_idx_dsl = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_cell_blk = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_distv_bary_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_distv_bary_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_dthalf = 2.0

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            cell_idx=cell_idx,
            cell_blk=cell_blk,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
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
