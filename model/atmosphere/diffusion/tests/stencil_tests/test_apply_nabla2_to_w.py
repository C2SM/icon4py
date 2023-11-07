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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import apply_nabla2_to_w
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoApplyNabla2ToW(StencilTest):
    PROGRAM = apply_nabla2_to_w
    OUTPUTS = ("w",)

    @staticmethod
    def reference(
        grid,
        area: np.array,
        z_nabla2_c: np.array,
        geofac_n2s: np.array,
        w: np.array,
        diff_multfac_w: float,
        **kwargs,
    ) -> np.array:
        c2e2cO = grid.connectivities[C2E2CODim]
        geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
        area = np.expand_dims(area, axis=-1)
        w = w - diff_multfac_w * area * area * np.sum(
            np.where((c2e2cO != -1)[:, :, np.newaxis], z_nabla2_c[c2e2cO] * geofac_n2s, 0.0), axis=1
        )
        return dict(w=w)

    @pytest.fixture
    def input_data(self, grid):
        area = random_field(grid, CellDim)
        z_nabla2_c = random_field(grid, CellDim, KDim)
        geofac_n2s = random_field(grid, CellDim, C2E2CODim)
        w = random_field(grid, CellDim, KDim)
        return dict(
            area=area,
            z_nabla2_c=z_nabla2_c,
            geofac_n2s=geofac_n2s,
            w=w,
            diff_multfac_w=5.0,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
