# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.integrate_tracer_horizontally import (
    integrate_tracer_horizontally,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


class TestIntegrateTracerHorizontally(stencil_tests.StencilTest):
    PROGRAM = integrate_tracer_horizontally
    OUTPUTS = ("tracer_new_hor",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        p_mflx_tracer_h: np.ndarray,
        deepatmo_divh: np.ndarray,
        tracer_now: np.ndarray,
        rhodz_now: np.ndarray,
        rhodz_new: np.ndarray,
        geofac_div: np.ndarray,
        p_dtime: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        tracer_new_hor = (
            tracer_now * rhodz_now
            - p_dtime
            * deepatmo_divh
            * np.sum(p_mflx_tracer_h[connectivities[dims.C2E]] * geofac_div, axis=1)
        ) / rhodz_new
        return dict(tracer_new_hor=tracer_new_hor)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_mflx_tracer_h = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        deepatmo_divh = self.data_alloc.random_field(dims.KDim)
        tracer_now = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_now = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_new = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        geofac_div = self.data_alloc.random_field(dims.CellDim, dims.C2EDim)
        p_dtime = np.float64(5.0)
        tracer_new_hor = self.data_alloc.zero_field(dims.CellDim, dims.KDim)
        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            deepatmo_divh=deepatmo_divh,
            tracer_now=tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=geofac_div,
            p_dtime=p_dtime,
            tracer_new_hor=tracer_new_hor,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
