# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_inverse_density_to_edges import (
    interpolate_inverse_density_to_edges,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestInterpolateInverseDensityToEdges(stencil_tests.StencilTest):
    PROGRAM = interpolate_inverse_density_to_edges
    OUTPUTS = ("inv_rhoe",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho: np.ndarray,
        c_lin_e: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]
        inv_rhoe = 1.0 / np.sum(rho[e2c] * c_lin_e[:, :, np.newaxis], axis=1)

        inv_rhoe_out = np.zeros_like(inv_rhoe)
        inv_rhoe_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = inv_rhoe[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(inv_rhoe=inv_rhoe_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        rho = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.5, high=2.0, dtype=ta.wpfloat
        )
        c_lin_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, low=0.1, high=0.9, dtype=ta.wpfloat
        )
        inv_rhoe = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran: cells2edges_scalar + reciprocal loop on
        # rl_start = grf_bdywidth_e + 1, rl_end = min_rledge_int.
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            rho=rho,
            c_lin_e=c_lin_e,
            inv_rhoe=inv_rhoe,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
