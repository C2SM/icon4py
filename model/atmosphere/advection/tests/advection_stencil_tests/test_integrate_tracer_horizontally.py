# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as xp
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_horizontally import (
    integrate_tracer_horizontally,
)
from icon4py.model.common import dimension as dims


class TestIntegrateTracerHorizontally(helpers.StencilTest):
    PROGRAM = integrate_tracer_horizontally
    OUTPUTS = ("tracer_new_hor",)

    @staticmethod
    def reference(
        grid,
        p_mflx_tracer_h: xp.array,
        deepatmo_divh: xp.array,
        tracer_now: xp.array,
        rhodz_now: xp.array,
        rhodz_new: xp.array,
        geofac_div: xp.array,
        p_dtime,
        **kwargs,
    ) -> dict:
        geofac_div = helpers.reshape(geofac_div, grid.connectivities[dims.C2EDim].shape)
        geofac_div = xp.expand_dims(xp.asarray(geofac_div), axis=-1)
        tracer_new_hor = (
            tracer_now * rhodz_now
            - p_dtime
            * deepatmo_divh
            * xp.sum(p_mflx_tracer_h[grid.connectivities[dims.C2EDim]] * geofac_div, axis=1)
        ) / rhodz_new
        return dict(tracer_new_hor=tracer_new_hor)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_mflx_tracer_h = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        deepatmo_divh = helpers.random_field(grid, dims.KDim)
        tracer_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        rhodz_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        rhodz_new = helpers.random_field(grid, dims.CellDim, dims.KDim)
        geofac_div = helpers.random_field(grid, dims.CellDim, dims.C2EDim)
        geofac_div_new = helpers.as_1D_sparse_field(geofac_div, dims.CEDim)
        p_dtime = xp.float64(5.0)
        tracer_new_hor = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            deepatmo_divh=deepatmo_divh,
            tracer_now=tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=geofac_div_new,
            p_dtime=p_dtime,
            tracer_new_hor=tracer_new_hor,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
