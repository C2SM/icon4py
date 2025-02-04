# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import ice_mass

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestIceNumber(StencilTest):
    PROGRAM = ice_mass
    OUTPUTS = ("ice_mass",)

    @staticmethod
    def reference(grid, qi: np.array, ni: np.array, **kwargs) -> dict:
        return dict(ice_mass=np.full(qi.shape, 1.0e-12))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qi              = constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            ni              = constant_field(grid, 5.05089, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_mass        = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
