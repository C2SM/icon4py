# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest

from icon4py.model.atmosphere.advection.hflx_limiter_mo_stencil_03 import (
    hflx_limiter_mo_stencil_03,
    hflx_limiter_mo_stencil_03_min_max,
)
from icon4py.model.common.dimension import C2E2CDim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestHflxLimiterMoStencil03MinMax(StencilTest):
    PROGRAM = hflx_limiter_mo_stencil_03_min_max
    OUTPUTS = ("z_max", "z_min")

    @staticmethod
    def reference(grid, z_tracer_max, z_tracer_min, beta_fct, r_beta_fct, **kwargs):
        c2e2c = grid.connectivities[C2E2CDim]
        z_max = beta_fct * np.maximum(np.max(z_tracer_max[c2e2c], axis=1), z_tracer_max)
        z_min = r_beta_fct * np.minimum(np.min(z_tracer_min[c2e2c], axis=1), z_tracer_min)
        return dict(z_max=z_max, z_min=z_min)

    @pytest.fixture
    def input_data(self, grid):
        z_tracer_max = random_field(grid, CellDim, KDim)
        z_tracer_min = random_field(grid, CellDim, KDim)
        beta_fct = 0.9
        r_beta_fct = 0.3
        z_max = zero_field(grid, CellDim, KDim)
        z_min = zero_field(grid, CellDim, KDim)
        return dict(
            z_tracer_max=z_tracer_max,
            z_tracer_min=z_tracer_min,
            beta_fct=beta_fct,
            r_beta_fct=r_beta_fct,
            z_max=z_max,
            z_min=z_min,
        )


class TestHflxLimiterMoStencil03(StencilTest):
    PROGRAM = hflx_limiter_mo_stencil_03
    OUTPUTS = ("r_p", "r_m")

    @staticmethod
    def reference(
        grid,
        z_tracer_max,
        z_tracer_min,
        beta_fct,
        r_beta_fct,
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        dbl_eps,
        **kwargs,
    ):
        res = TestHflxLimiterMoStencil03MinMax.reference(
            grid, z_tracer_max, z_tracer_min, beta_fct, r_beta_fct
        )
        z_max, z_min = res["z_max"], res["z_min"]
        r_p = (z_max - z_tracer_new_low) / (z_mflx_anti_in + dbl_eps)
        r_m = (z_tracer_new_low - z_min) / (z_mflx_anti_out + dbl_eps)
        return dict(r_p=r_p, r_m=r_m)

    @pytest.fixture
    def input_data(self, grid):
        z_tracer_max = random_field(grid, CellDim, KDim)
        z_tracer_min = random_field(grid, CellDim, KDim)
        beta_fct = 0.4
        r_beta_fct = 0.6
        z_mflx_anti_in = random_field(grid, CellDim, KDim)
        z_mflx_anti_out = random_field(grid, CellDim, KDim)
        z_tracer_new_low = random_field(grid, CellDim, KDim)
        dbl_eps = 1e-5
        r_p = zero_field(grid, CellDim, KDim)
        r_m = zero_field(grid, CellDim, KDim)
        return dict(
            z_tracer_max=z_tracer_max,
            z_tracer_min=z_tracer_min,
            beta_fct=beta_fct,
            r_beta_fct=r_beta_fct,
            z_mflx_anti_in=z_mflx_anti_in,
            z_mflx_anti_out=z_mflx_anti_out,
            z_tracer_new_low=z_tracer_new_low,
            dbl_eps=dbl_eps,
            r_p=r_p,
            r_m=r_m,
        )
