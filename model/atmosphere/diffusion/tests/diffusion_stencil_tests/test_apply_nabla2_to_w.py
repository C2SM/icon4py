# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import apply_nabla2_to_w
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def apply_nabla2_to_w_numpy(
    grid,
    area: xp.array,
    z_nabla2_c: xp.array,
    geofac_n2s: xp.array,
    w: xp.array,
    diff_multfac_w: float,
) -> xp.array:
    c2e2cO = grid.connectivities[dims.C2E2CODim]
    geofac_n2s = xp.expand_dims(geofac_n2s, axis=-1)
    area = xp.expand_dims(area, axis=-1)
    w = w - diff_multfac_w * area * area * xp.sum(
        xp.where((c2e2cO != -1)[:, :, xp.newaxis], z_nabla2_c[c2e2cO] * geofac_n2s, 0.0), axis=1
    )
    return w


class TestMoApplyNabla2ToW(StencilTest):
    PROGRAM = apply_nabla2_to_w
    OUTPUTS = ("w",)

    @staticmethod
    def reference(
        grid,
        area: xp.array,
        z_nabla2_c: xp.array,
        geofac_n2s: xp.array,
        w: xp.array,
        diff_multfac_w: float,
        **kwargs,
    ) -> dict:
        w = apply_nabla2_to_w_numpy(grid, area, z_nabla2_c, geofac_n2s, w, diff_multfac_w)
        return dict(w=w)

    @pytest.fixture
    def input_data(self, grid):
        area = random_field(grid, dims.CellDim, dtype=wpfloat)
        z_nabla2_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        geofac_n2s = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        return dict(
            area=area,
            z_nabla2_c=z_nabla2_c,
            geofac_n2s=geofac_n2s,
            w=w,
            diff_multfac_w=wpfloat("5.0"),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
