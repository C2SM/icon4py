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

import icon4py.model.testing.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.apply_vertical_density_increment import (
    apply_vertical_density_increment,
)
from icon4py.model.common import dimension as dims


class TestApplyVerticalDensityIncrement(helpers.StencilTest):
    PROGRAM = apply_vertical_density_increment
    OUTPUTS = (
        helpers.Output(
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
    ) -> dict:
        tmp = p_dtime * (
            p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
        )
        rhodz_ast2 = rhodz_ast + tmp

        return dict(rhodz_ast2=rhodz_ast2)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        rhodz_ast = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_contra_v = helpers.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        deepatmo_divzl = helpers.random_field(grid, dims.KDim)
        deepatmo_divzu = helpers.random_field(grid, dims.KDim)
        rhodz_ast2 = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = 0.1
        return dict(
            rhodz_ast=rhodz_ast,
            p_mflx_contra_v=p_mflx_contra_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_ast2=rhodz_ast2,
            p_dtime=p_dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
