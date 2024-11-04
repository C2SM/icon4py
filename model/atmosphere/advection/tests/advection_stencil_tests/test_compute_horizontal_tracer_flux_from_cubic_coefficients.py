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
from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_cubic_coefficients import (
    compute_horizontal_tracer_flux_from_cubic_coefficients,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp


class TestComputeHorizontalTracerFluxFromCubicCoefficients(helpers.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_from_cubic_coefficients
    OUTPUTS = ("p_out_e_hybrid_2",)

    @staticmethod
    def reference(
        grid,
        p_out_e_hybrid_2: xp.array,
        p_mass_flx_e: xp.array,
        z_dreg_area: xp.array,
        **kwargs,
    ) -> dict:
        p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

        return dict(p_out_e_hybrid_2=p_out_e_hybrid_2)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_out_e_hybrid_2 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_mass_flx_e = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        z_dreg_area = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            p_mass_flx_e=p_mass_flx_e,
            z_dreg_area=z_dreg_area,
            p_out_e_hybrid_2=p_out_e_hybrid_2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )