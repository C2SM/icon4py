# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, base as base_grid
from icon4py.model.common.math.stencils.compute_nabla2_on_cell import compute_nabla2_on_cell
from icon4py.model.common.math.stencils.compute_nabla2_on_cell_k import compute_nabla2_on_cell_k
from icon4py.model.testing import reference_funcs, stencil_tests
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager


@pytest.mark.embedded_remap_error
class TestNabla2OnCell(stencil_tests.StencilTest):
    PROGRAM = compute_nabla2_on_cell
    OUTPUTS = ("nabla2_psi_c",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        psi_c: np.ndarray,
        geofac_n2s: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        nabla2_psi_c_np = reference_funcs.nabla2_on_cell_numpy(connectivities, psi_c, geofac_n2s)
        return dict(nabla2_psi_c=nabla2_psi_c_np)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        psi_c = self.data_alloc.constant_field(1.0, dims.CellDim)
        geofac_n2s = self.data_alloc.constant_field(2.0, dims.CellDim, dims.C2E2CODim)
        nabla2_psi_c = self.data_alloc.zero_field(dims.CellDim)
        return dict(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
        )


@pytest.mark.embedded_remap_error
class TestNabla2OnCellK(stencil_tests.StencilTest):
    PROGRAM = compute_nabla2_on_cell_k
    OUTPUTS = ("nabla2_psi_c",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        psi_c: np.ndarray,
        geofac_n2s: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        nabla2_psi_c_np = reference_funcs.nabla2_on_cell_k_numpy(connectivities, psi_c, geofac_n2s)
        return dict(nabla2_psi_c=nabla2_psi_c_np)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        psi_c = self.data_alloc.constant_field(1.0, dims.CellDim, dims.KDim)
        geofac_n2s = self.data_alloc.constant_field(2.0, dims.CellDim, dims.C2E2CODim)
        nabla2_psi_c = self.data_alloc.zero_field(dims.CellDim, dims.KDim)
        return dict(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
