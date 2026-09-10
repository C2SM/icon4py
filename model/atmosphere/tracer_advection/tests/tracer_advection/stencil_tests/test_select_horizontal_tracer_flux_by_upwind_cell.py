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
from icon4py.model.atmosphere.tracer_advection.stencils.select_horizontal_tracer_flux_by_upwind_cell import (
    select_horizontal_tracer_flux_by_upwind_cell,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


class TestSelectHorizontalTracerFluxByUpwindCell(stencil_tests.StencilTest):
    PROGRAM = select_horizontal_tracer_flux_by_upwind_cell
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        use_first: np.ndarray,
        p_cell_rel_idx_dsl: np.ndarray,
        p_flux_first: np.ndarray,
        p_flux_second: np.ndarray,
        **kwargs,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]
        use_first_upwind = np.where(
            p_cell_rel_idx_dsl == 1, use_first[e2c[:, 1]], use_first[e2c[:, 0]]
        )
        return dict(p_out_e=np.where(use_first_upwind, p_flux_first, p_flux_second))

    @pytest.fixture
    def input_data(self, grid) -> dict:
        rel_idx = gtx.as_field(
            (dims.EdgeDim, dims.KDim),
            (np.add.outer(np.arange(grid.num_edges), np.arange(grid.num_levels)) % 2).astype(
                gtx.int32
            ),
        )
        return dict(
            use_first=data_alloc.random_mask(grid, dims.CellDim, dims.KDim),
            p_cell_rel_idx_dsl=rel_idx,
            p_flux_first=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_flux_second=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_out_e=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
