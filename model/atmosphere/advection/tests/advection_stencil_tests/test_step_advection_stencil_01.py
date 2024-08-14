# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.step_advection_stencil_01 import step_advection_stencil_01
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import Output, StencilTest, random_field, zero_field


class TestStepAdvectionStencil01(StencilTest):
    PROGRAM = step_advection_stencil_01
    OUTPUTS = (
        Output(
            "rhodz_ast2",
            refslice=(slice(None), slice(None, -1)),
            gtslice=(slice(None), slice(None, -1)),
        ),
    )

    @staticmethod
    def reference(
        grid,
        rhodz_ast: np.array,
        p_mflx_contra_v: np.array,
        deepatmo_divzl: np.array,
        deepatmo_divzu: np.array,
        p_dtime,
        **kwargs,
    ):
        tmp = p_dtime * (
            p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
        )
        rhodz_ast2 = rhodz_ast + tmp

        return dict(rhodz_ast2=rhodz_ast2)

    @pytest.fixture
    def input_data(self, grid):
        rhodz_ast = random_field(grid, CellDim, KDim)
        p_mflx_contra_v = random_field(grid, CellDim, KDim, extend={KDim: 1})
        deepatmo_divzl = random_field(grid, KDim)
        deepatmo_divzu = random_field(grid, KDim)
        rhodz_ast2 = zero_field(grid, CellDim, KDim)
        p_dtime = 0.1
        return dict(
            rhodz_ast=rhodz_ast,
            p_mflx_contra_v=p_mflx_contra_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            p_dtime=p_dtime,
            rhodz_ast2=rhodz_ast2,
        )
