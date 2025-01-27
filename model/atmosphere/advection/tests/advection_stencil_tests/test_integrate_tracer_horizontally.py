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
import icon4py.model.testing.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_horizontally import (
    integrate_tracer_horizontally,
)
from icon4py.model.common import dimension as dims


class TestIntegrateTracerHorizontally(helpers.StencilTest):
    PROGRAM = integrate_tracer_horizontally
    OUTPUTS = ("tracer_new_hor",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        p_mflx_tracer_h: np.array,
        deepatmo_divh: np.array,
        tracer_now: np.array,
        rhodz_now: np.array,
        rhodz_new: np.array,
        geofac_div: np.array,
        p_dtime,
        **kwargs,
    ) -> dict:
        geofac_div = helpers.reshape(geofac_div, connectivities[dims.C2EDim].shape)
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        tracer_new_hor = (
            tracer_now * rhodz_now
            - p_dtime
            * deepatmo_divh
            * np.sum(p_mflx_tracer_h[connectivities[dims.C2EDim]] * geofac_div, axis=1)
        ) / rhodz_new
        return dict(tracer_new_hor=tracer_new_hor)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_mflx_tracer_h = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        deepatmo_divh = data_alloc.random_field(grid, dims.KDim)
        tracer_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rhodz_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rhodz_new = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        geofac_div = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim)
        geofac_div_new = data_alloc.as_1D_sparse_field(geofac_div, dims.CEDim)
        p_dtime = np.float64(5.0)
        tracer_new_hor = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
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
