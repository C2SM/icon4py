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

from icon4py.model.atmosphere.advection.hflx_limiter_mo_stencil_01b import (
    hflx_limiter_mo_stencil_01b,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    reshape,
    zero_field,
)


class TestHflxLimiterMoStencil01b(StencilTest):
    PROGRAM = hflx_limiter_mo_stencil_01b
    OUTPUTS = (
        "z_mflx_anti_in",
        "z_mflx_anti_out",
        "z_tracer_new_low",
        "z_tracer_max",
        "z_tracer_min",
    )

    @staticmethod
    def reference(
        grid,
        geofac_div: np.ndarray,
        p_rhodz_now: np.ndarray,
        p_rhodz_new: np.ndarray,
        z_mflx_low: np.ndarray,
        z_anti: np.ndarray,
        p_cc: np.ndarray,
        p_dtime: float,
        **kwargs,
    ):
        c2e = grid.connectivities[C2EDim]
        z_anti_c2e = z_anti[c2e]

        geofac_div = reshape(geofac_div, c2e.shape)
        geofac_div = np.expand_dims(geofac_div, axis=-1)

        zero_array = np.zeros(p_rhodz_now.shape)

        z_mflx_anti_1 = p_dtime * geofac_div[:, 0] / p_rhodz_new * z_anti_c2e[:, 0]
        z_mflx_anti_2 = p_dtime * geofac_div[:, 1] / p_rhodz_new * z_anti_c2e[:, 1]
        z_mflx_anti_3 = p_dtime * geofac_div[:, 2] / p_rhodz_new * z_anti_c2e[:, 2]

        z_mflx_anti_in = -1.0 * (
            np.minimum(zero_array, z_mflx_anti_1)
            + np.minimum(zero_array, z_mflx_anti_2)
            + np.minimum(zero_array, z_mflx_anti_3)
        )

        z_mflx_anti_out = (
            np.maximum(zero_array, z_mflx_anti_1)
            + np.maximum(zero_array, z_mflx_anti_2)
            + np.maximum(zero_array, z_mflx_anti_3)
        )

        z_fluxdiv_c = np.sum(z_mflx_low[c2e] * geofac_div, axis=1)

        z_tracer_new_low = (p_cc * p_rhodz_now - p_dtime * z_fluxdiv_c) / p_rhodz_new
        z_tracer_max = np.maximum(p_cc, z_tracer_new_low)
        z_tracer_min = np.minimum(p_cc, z_tracer_new_low)

        return dict(
            z_mflx_anti_in=z_mflx_anti_in,
            z_mflx_anti_out=z_mflx_anti_out,
            z_tracer_new_low=z_tracer_new_low,
            z_tracer_max=z_tracer_max,
            z_tracer_min=z_tracer_min,
        )

    @pytest.fixture
    def input_data(self, grid):
        geofac_div = random_field(grid, CellDim, C2EDim)
        geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)
        p_rhodz_now = random_field(grid, CellDim, KDim)
        p_rhodz_new = random_field(grid, CellDim, KDim)
        z_mflx_low = random_field(grid, EdgeDim, KDim)
        z_anti = random_field(grid, EdgeDim, KDim)
        p_cc = random_field(grid, CellDim, KDim)
        p_dtime = 5.0

        z_mflx_anti_in = zero_field(grid, CellDim, KDim)
        z_mflx_anti_out = zero_field(grid, CellDim, KDim)
        z_tracer_new_low = zero_field(grid, CellDim, KDim)
        z_tracer_max = zero_field(grid, CellDim, KDim)
        z_tracer_min = zero_field(grid, CellDim, KDim)

        return dict(
            geofac_div=geofac_div_new,
            p_rhodz_now=p_rhodz_now,
            p_rhodz_new=p_rhodz_new,
            z_mflx_low=z_mflx_low,
            z_anti=z_anti,
            p_cc=p_cc,
            p_dtime=p_dtime,
            z_mflx_anti_in=z_mflx_anti_in,
            z_mflx_anti_out=z_mflx_anti_out,
            z_tracer_new_low=z_tracer_new_low,
            z_tracer_max=z_tracer_max,
            z_tracer_min=z_tracer_min,
        )
