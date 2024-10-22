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

from icon4py.model.atmosphere.dycore.compute_avg_vn import compute_avg_vn
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestComputeAvgVn(StencilTest):
    PROGRAM = compute_avg_vn
    OUTPUTS = ("z_vn_avg",)

    @staticmethod
    def reference(grid, e_flx_avg: np.array, vn: np.array, **kwargs) -> dict:
        e2c2eO = grid.connectivities[dims.E2C2EODim]
        geofac_grdiv = np.expand_dims(e_flx_avg, axis=-1)
        z_vn_avg = np.sum(
            np.where((e2c2eO != -1)[:, :, np.newaxis], vn[e2c2eO] * geofac_grdiv, 0), axis=1
        )
        return dict(z_vn_avg=z_vn_avg)

    @pytest.fixture
    def input_data(self, grid):
        e_flx_avg = random_field(grid, dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_vn_avg = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            e_flx_avg=e_flx_avg,
            vn=vn,
            z_vn_avg=z_vn_avg,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
