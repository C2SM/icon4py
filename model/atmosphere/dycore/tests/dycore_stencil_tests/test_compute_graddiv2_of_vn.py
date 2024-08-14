# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.compute_graddiv2_of_vn import compute_graddiv2_of_vn
from icon4py.model.common.dimension import E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeGraddiv2OfVn(StencilTest):
    PROGRAM = compute_graddiv2_of_vn
    OUTPUTS = ("z_graddiv2_vn",)

    @staticmethod
    def reference(grid, geofac_grdiv: np.array, z_graddiv_vn: np.array, **kwargs) -> dict:
        e2c2eO = grid.connectivities[E2C2EODim]
        geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
        z_graddiv2_vn = np.sum(
            np.where((e2c2eO != -1)[:, :, np.newaxis], z_graddiv_vn[e2c2eO] * geofac_grdiv, 0),
            axis=1,
        )
        return dict(z_graddiv2_vn=z_graddiv2_vn)

    @pytest.fixture
    def input_data(self, grid):
        z_graddiv_vn = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        geofac_grdiv = random_field(grid, EdgeDim, E2C2EODim, dtype=wpfloat)
        z_graddiv2_vn = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            geofac_grdiv=geofac_grdiv,
            z_graddiv_vn=z_graddiv_vn,
            z_graddiv2_vn=z_graddiv2_vn,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
