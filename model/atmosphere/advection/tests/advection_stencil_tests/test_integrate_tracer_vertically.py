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
from gt4py.next import as_field

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_vertically import (
    integrate_tracer_vertically,
)
from icon4py.model.common import dimension as dims


class TestIntegrateTracerVertically(helpers.StencilTest):
    PROGRAM = integrate_tracer_vertically
    OUTPUTS = ("tracer_new",)

    @staticmethod
    def reference(
        grid,
        tracer_now: np.array,
        rhodz_now: np.array,
        p_mflx_tracer_v: np.array,
        deepatmo_divzl: np.array,
        deepatmo_divzu: np.array,
        rhodz_new: np.array,
        k: np.array,
        ivadv_tracer: gtx.int32,
        iadv_slev_jt: gtx.int32,
        p_dtime: float,
        **kwargs,
    ) -> np.array:
        if ivadv_tracer != 0:
            tracer_new = np.where(
                (iadv_slev_jt <= k),
                (
                    tracer_now * rhodz_now
                    + p_dtime
                    * (
                        p_mflx_tracer_v[:, 1:] * deepatmo_divzl
                        - p_mflx_tracer_v[:, :-1] * deepatmo_divzu
                    )
                )
                / rhodz_new,
                tracer_now,
            )
        else:
            tracer_new = tracer_now

        return dict(tracer_new=tracer_new)

    @pytest.fixture
    def input_data(self, grid):
        tracer_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        rhodz_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_tracer_v = helpers.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        deepatmo_divzl = helpers.random_field(grid, dims.KDim)
        deepatmo_divzu = helpers.random_field(grid, dims.KDim)
        rhodz_new = helpers.random_field(grid, dims.CellDim, dims.KDim)
        k = as_field((dims.KDim,), np.arange(grid.num_levels, dtype=gtx.int32))
        p_dtime = np.float64(5.0)
        ivadv_tracer = 1
        iadv_slev_jt = 4
        tracer_new = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            tracer_now=tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_new=rhodz_new,
            k=k,
            p_dtime=p_dtime,
            ivadv_tracer=ivadv_tracer,
            iadv_slev_jt=iadv_slev_jt,
            tracer_new=tracer_new,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )