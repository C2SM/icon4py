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
from icon4py.model.atmosphere.tracer_advection.stencils.apply_cell_local_positive_definite_horizontal_flux_factor import (
    apply_cell_local_positive_definite_horizontal_flux_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


class TestApplyCellLocalPositiveDefiniteHorizontalFluxFactor(stencil_tests.StencilTest):
    PROGRAM = apply_cell_local_positive_definite_horizontal_flux_factor
    OUTPUTS = ("p_mflx_tracer_h",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        r_m: np.ndarray,
        p_mflx_tracer_h: np.ndarray,
        p_vn: np.ndarray,
        **kwargs,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]
        lvn_pos = p_vn >= 0.0
        orientation = np.where(lvn_pos, 1.0, -1.0)
        z_b = orientation * np.maximum(orientation * p_mflx_tracer_h, 0.0)
        r_m_upwind = np.where(lvn_pos, r_m[e2c[:, 0]], r_m[e2c[:, 1]])
        return dict(p_mflx_tracer_h=r_m_upwind * z_b)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        return dict(
            r_m=data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=0.0, high=1.0),
            p_mflx_tracer_h=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_vn=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
