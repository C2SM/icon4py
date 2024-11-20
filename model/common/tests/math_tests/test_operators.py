# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.math.stencils.compute_nabla2_on_cell import compute_nabla2_on_cell
from icon4py.model.common.math.stencils.compute_nabla2_on_cell_k import compute_nabla2_on_cell_k
from icon4py.model.common.test_utils import helpers as test_helpers, reference_funcs
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field


class TestNabla2OnCell(StencilTest):
    PROGRAM = compute_nabla2_on_cell
    OUTPUTS = ("nabla2_psi_c",)

    @staticmethod
    def reference(
        grid,
        psi_c: np.array,
        geofac_n2s: np.array,
        **kwargs,
    ) -> dict:
        nabla2_psi_c_np = reference_funcs.nabla2_on_cell_numpy(
            grid, psi_c, geofac_n2s
        )
        return dict(nabla2_psi_c=nabla2_psi_c_np)

    @pytest.fixture
    def input_data(self, grid):
        psi_c = constant_field(grid, 1.0, dims.CellDim)
        geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
        nabla2_psi_c = zero_field(grid, dims.CellDim)
        return dict(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
        )

class TestNabla2OnCellK(StencilTest):
    PROGRAM = compute_nabla2_on_cell_k
    OUTPUTS = ("nabla2_psi_c",)

    @staticmethod
    def reference(
        grid,
        psi_c: np.array,
        geofac_n2s: np.array,
        **kwargs,
    ) -> dict:
        nabla2_psi_c_np = reference_funcs.nabla2_on_cell_k_numpy(
            grid, psi_c, geofac_n2s
        )
        return dict(nabla2_psi_c=nabla2_psi_c_np)

    @pytest.fixture
    def input_data(self, grid):
        psi_c = constant_field(grid, 1.0, dims.CellDim, dims.KDim)
        geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
        nabla2_psi_c = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )

