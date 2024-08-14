# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.hor_adv_stencil_01 import hor_adv_stencil_01
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    reshape,
    zero_field,
)


class TestHorAdvStencil01(StencilTest):
    PROGRAM = hor_adv_stencil_01
    OUTPUTS = ("tracer_new_hor",)

    @staticmethod
    def reference(
        grid,
        p_mflx_tracer_h: np.array,
        deepatmo_divh: np.array,
        tracer_now: np.array,
        rhodz_now: np.array,
        rhodz_new: np.array,
        geofac_div: np.array,
        p_dtime,
        **kwargs,
    ) -> np.array:
        geofac_div = reshape(geofac_div, grid.connectivities[C2EDim].shape)
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        tracer_new_hor = (
            tracer_now * rhodz_now
            - p_dtime
            * deepatmo_divh
            * np.sum(p_mflx_tracer_h[grid.connectivities[C2EDim]] * geofac_div, axis=1)
        ) / rhodz_new
        return dict(tracer_new_hor=tracer_new_hor)

    @pytest.fixture
    def input_data(self, grid):
        p_mflx_tracer_h = random_field(grid, EdgeDim, KDim)
        deepatmo_divh = random_field(grid, KDim)
        tracer_now = random_field(grid, CellDim, KDim)
        rhodz_now = random_field(grid, CellDim, KDim)
        rhodz_new = random_field(grid, CellDim, KDim)
        geofac_div = random_field(grid, CellDim, C2EDim)
        geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)
        p_dtime = np.float64(5.0)
        tracer_new_hor = zero_field(grid, CellDim, KDim)
        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            deepatmo_divh=deepatmo_divh,
            tracer_now=tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=geofac_div_new,
            p_dtime=p_dtime,
            tracer_new_hor=tracer_new_hor,
        )
