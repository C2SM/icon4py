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
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest

class TestDepositionAutoConversion(StencilTest):
    PROGRAM = deposition_auto_conversion
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, qi: np.array, m_ice: np.array, ice_dep: np.array, **kwargs) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 6.6430804299795412e-08))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qi              = data_alloc.constant_field(grid, 2.02422e-2 + graupel_ct.qmin, dims.CellDim, dims.KDim, dtype=wpfloat),
            m_ice           = data_alloc.constant_field(grid, 1.0e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_dep         = data_alloc.constant_field(grid, 2.06276e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            conversion_rate = data_alloc.constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
