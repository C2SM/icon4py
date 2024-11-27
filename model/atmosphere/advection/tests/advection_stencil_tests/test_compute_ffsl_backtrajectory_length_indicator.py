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
from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory_length_indicator import (
    compute_ffsl_backtrajectory_length_indicator,
)
from icon4py.model.common import dimension as dims
import numpy as xp


class TestComputeFfslBacktrajectoryLengthIndicator(helpers.StencilTest):
    PROGRAM = compute_ffsl_backtrajectory_length_indicator
    OUTPUTS = ("opt_famask_dsl",)

    @staticmethod
    def reference(grid, p_vn: xp.array, p_vt: xp.array, p_dt: float, **kwargs) -> dict:
        lvn_pos = p_vn >= 0.0

        traj_length = xp.sqrt(p_vn**2 + p_vt**2) * p_dt

        edge_cell_length = xp.expand_dims(
            xp.asarray(grid.connectivities[dims.E2CDim], dtype=float), axis=-1
        )
        e2c_length = xp.where(lvn_pos, edge_cell_length[:, 0], edge_cell_length[:, 1])

        opt_famask_dsl = xp.where(
            traj_length > (1.25 * xp.broadcast_to(e2c_length, p_vn.shape)),
            1,
            0,
        )

        return dict(opt_famask_dsl=opt_famask_dsl)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        edge_cell_length = xp.asarray(grid.connectivities[dims.E2CDim], dtype=float)
        edge_cell_length_new = helpers.numpy_to_1D_sparse_field(edge_cell_length, dims.ECDim)
        opt_famask_dsl = helpers.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_dt = 1.0

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            edge_cell_length=edge_cell_length_new,
            opt_famask_dsl=opt_famask_dsl,
            p_dt=p_dt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
