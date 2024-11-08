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
from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_upwind import (
    compute_horizontal_tracer_flux_upwind,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp


class TestComputeHorizontalTracerFluxUpwind(helpers.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_upwind
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        grid,
        p_cc: xp.array,
        p_mass_flx_e: xp.array,
        p_vn: xp.array,
        **kwargs,
    ) -> dict:
        e2c = grid.connectivities[dims.E2CDim]
        p_out_e = xp.where(p_vn > 0.0, p_cc[e2c][:, 0], p_cc[e2c][:, 1]) * p_mass_flx_e
        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_mass_flx_e = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_out_e = helpers.zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            p_cc=p_cc,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
            p_out_e=p_out_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
