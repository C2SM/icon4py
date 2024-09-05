# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_cubic_coefficients import (
    compute_horizontal_tracer_flux_from_cubic_coefficients,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestComputeHorizontalTracerFluxFromCubicCoefficients(StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_from_cubic_coefficients
    OUTPUTS = ("p_out_e_hybrid_2",)

    @staticmethod
    def reference(
        grid,
        p_out_e_hybrid_2: np.array,
        p_mass_flx_e: np.array,
        z_dreg_area: np.array,
        **kwargs,
    ):
        p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

        return dict(p_out_e_hybrid_2=p_out_e_hybrid_2)

    @pytest.fixture
    def input_data(self, grid):
        p_out_e_hybrid_2 = random_field(grid, dims.EdgeDim, dims.KDim)
        p_mass_flx_e = random_field(grid, dims.EdgeDim, dims.KDim)
        z_dreg_area = random_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            p_mass_flx_e=p_mass_flx_e,
            z_dreg_area=z_dreg_area,
            p_out_e_hybrid_2=p_out_e_hybrid_2,
        )
