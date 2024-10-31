# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    calculate_nabla2_for_w,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field


def calculate_nabla2_for_w_numpy(grid, w: xp.array, geofac_n2s: xp.array):
    c2e2cO = xp.asarray(grid.connectivities[dims.C2E2CODim])
    geofac_n2s = xp.expand_dims(xp.asarray(geofac_n2s), axis=-1)
    z_nabla2_c = xp.sum(
        xp.where((c2e2cO != -1)[:, :, xp.newaxis], w[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return z_nabla2_c


class TestCalculateNabla2ForW(StencilTest):
    PROGRAM = calculate_nabla2_for_w
    OUTPUTS = ("z_nabla2_c",)

    @staticmethod
    def reference(grid, w: xp.array, geofac_n2s: xp.array, **kwargs) -> dict:
        z_nabla2_c = calculate_nabla2_for_w_numpy(grid, w, geofac_n2s)
        return dict(z_nabla2_c=z_nabla2_c)

    @pytest.fixture
    def input_data(self, grid):
        w = constant_field(grid, 1.0, dims.CellDim, dims.KDim)
        geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
        z_nabla2_c = zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            w=w,
            geofac_n2s=geofac_n2s,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
