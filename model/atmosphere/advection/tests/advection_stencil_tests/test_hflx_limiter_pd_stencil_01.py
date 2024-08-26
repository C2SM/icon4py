# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.hflx_limiter_pd_stencil_01 import (
    hflx_limiter_pd_stencil_01,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    reshape,
    zero_field,
)


class TestHflxLimiterPdStencil01(StencilTest):
    PROGRAM = hflx_limiter_pd_stencil_01
    OUTPUTS = ("r_m",)

    @staticmethod
    def reference(
        grid,
        geofac_div: np.ndarray,
        p_cc: np.ndarray,
        p_rhodz_now: np.ndarray,
        p_mflx_tracer_h: np.ndarray,
        p_dtime,
        dbl_eps,
        **kwargs,
    ):
        geofac_div = reshape(geofac_div, grid.connectivities[C2EDim].shape)
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        p_m_0 = np.maximum(
            0.0, p_mflx_tracer_h[grid.connectivities[C2EDim][:, 0]] * geofac_div[:, 0] * p_dtime
        )
        p_m_1 = np.maximum(
            0.0, p_mflx_tracer_h[grid.connectivities[C2EDim][:, 1]] * geofac_div[:, 1] * p_dtime
        )
        p_m_2 = np.maximum(
            0.0, p_mflx_tracer_h[grid.connectivities[C2EDim][:, 2]] * geofac_div[:, 2] * p_dtime
        )

        p_m = p_m_0 + p_m_1 + p_m_2
        r_m = np.minimum(1.0, p_cc * p_rhodz_now / (p_m + dbl_eps))

        return dict(r_m=r_m)

    @pytest.fixture
    def input_data(self, grid):
        geofac_div = random_field(grid, CellDim, C2EDim)
        geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)
        p_cc = random_field(grid, CellDim, KDim)
        p_rhodz_now = random_field(grid, CellDim, KDim)
        p_mflx_tracer_h = random_field(grid, EdgeDim, KDim)
        r_m = zero_field(grid, CellDim, KDim)
        p_dtime = np.float64(5)
        dbl_eps = np.float64(1e-9)
        return dict(
            geofac_div=geofac_div_new,
            p_cc=p_cc,
            p_rhodz_now=p_rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_dtime=p_dtime,
            dbl_eps=dbl_eps,
            r_m=r_m,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
