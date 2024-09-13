# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory_length_indicator import (
    compute_ffsl_backtrajectory_length_indicator,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    numpy_to_1D_sparse_field,
    random_field,
    zero_field,
)


class TestComputeFfslBacktrajectoryLengthIndicator(StencilTest):
    PROGRAM = compute_ffsl_backtrajectory_length_indicator
    OUTPUTS = ("opt_famask_dsl",)

    @staticmethod
    def reference(grid, p_vn: np.array, p_vt: np.array, p_dt: float, **kwargs):
        lvn_pos = np.where(p_vn >= 0.0, True, False)

        traj_length = np.sqrt(p_vn**2 + p_vt**2) * p_dt

        edge_cell_length = np.expand_dims(
            np.asarray(grid.connectivities[dims.E2CDim], dtype=float), axis=-1
        )
        e2c_length = np.where(lvn_pos, edge_cell_length[:, 0], edge_cell_length[:, 1])

        opt_famask_dsl = np.where(
            traj_length > (1.25 * np.broadcast_to(e2c_length, p_vn.shape)),
            1,
            0,
        )

        return dict(opt_famask_dsl=opt_famask_dsl)

    @pytest.fixture
    def input_data(self, grid):
        p_vn = random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = random_field(grid, dims.EdgeDim, dims.KDim)
        edge_cell_length = np.asarray(grid.connectivities[dims.E2CDim], dtype=float)
        edge_cell_length_new = numpy_to_1D_sparse_field(edge_cell_length, dims.ECDim)
        p_dt = 1.0
        opt_famask_dsl = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=int32)

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            edge_cell_length=edge_cell_length_new,
            p_dt=p_dt,
            opt_famask_dsl=opt_famask_dsl,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
