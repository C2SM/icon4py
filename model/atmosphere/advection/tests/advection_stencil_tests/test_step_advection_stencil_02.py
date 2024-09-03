# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.step_advection_stencil_02 import (
    step_advection_stencil_02,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import Output, StencilTest, random_field, zero_field


class TestStepAdvectionStencil02(StencilTest):
    PROGRAM = step_advection_stencil_02
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
        p_rhodz_new = random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_contra_v = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        deepatmo_divzl = random_field(grid, dims.KDim)
        deepatmo_divzu = random_field(grid, dims.KDim)
        p_dtime = 0.1
        rhodz_ast2 = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            p_rhodz_new=p_rhodz_new,
            p_mflx_contra_v=p_mflx_contra_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            p_dtime=p_dtime,
            rhodz_ast2=rhodz_ast2,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
