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
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.vert_adv_stencil_01 import vert_adv_stencil_01
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestVertAdvStencil01(StencilTest):
    PROGRAM = vert_adv_stencil_01
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
        ivadv_tracer: int32,
        iadv_slev_jt: int32,
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
        tracer_now = random_field(grid, CellDim, KDim)
        rhodz_now = random_field(grid, CellDim, KDim)
        p_mflx_tracer_v = random_field(grid, CellDim, KDim, extend={KDim: 1})
        deepatmo_divzl = random_field(grid, KDim)
        deepatmo_divzu = random_field(grid, KDim)
        rhodz_new = random_field(grid, CellDim, KDim)
        k = as_field((KDim,), np.arange(grid.num_levels, dtype=int32))
        p_dtime = np.float64(5.0)
        ivadv_tracer = 1
        iadv_slev_jt = 4
        tracer_new = zero_field(grid, CellDim, KDim)
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
        )
