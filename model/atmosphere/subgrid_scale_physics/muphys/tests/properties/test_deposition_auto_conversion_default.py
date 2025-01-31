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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import deposition_auto_conversion

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestDepositionAutoConversionDefault(StencilTest):
    PROGRAM = deposition_auto_conversion
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, qi: np.array, m_ice: np.array, ice_dep: np.array, **kwargs) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qi              = constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            m_ice           = constant_field(grid, 1.0e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_dep         = constant_field(grid, -2.06276e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            conversion_rate = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
