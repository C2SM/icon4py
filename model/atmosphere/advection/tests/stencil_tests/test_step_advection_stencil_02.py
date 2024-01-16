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

from icon4py.model.atmosphere.advection.step_advection_stencil_02 import step_advection_stencil_02
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, Output, random_field, zero_field


class TestStepAdvectionStencil02(StencilTest):
    PROGRAM = step_advection_stencil_02
    OUTPUTS = (
        Output(
            "rhodz_ast2", refslice=(slice(None), slice(None, -1)), gtslice=(slice(None), slice(None, -1))
        ),
    )
    
    @staticmethod
    def reference(
        grid,
        p_rhodz_new: np.array,
        p_mflx_contra_v: np.array,
        deepatmo_divzl: np.array,
        deepatmo_divzu: np.array,
        p_dtime: float,
        **kwargs,
    ):
        tmp = p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
        rhodz_ast2 = np.maximum(0.1 * p_rhodz_new, p_rhodz_new) - p_dtime * tmp
        return dict(rhodz_ast2=rhodz_ast2)

    @pytest.fixture
    def input_data(self, grid):
        p_rhodz_new = random_field(grid, CellDim, KDim)
        p_mflx_contra_v = random_field(grid, CellDim, KDim, extend={KDim: 1})
        deepatmo_divzl = random_field(grid, KDim)
        deepatmo_divzu = random_field(grid, KDim)
        p_dtime = 0.1
        rhodz_ast2 = zero_field(grid, CellDim, KDim)
        return dict(
            p_rhodz_new=p_rhodz_new,
            p_mflx_contra_v=p_mflx_contra_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            p_dtime=p_dtime,
            rhodz_ast2=rhodz_ast2,
        )