# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import GraupelCt
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    deposition_auto_conversion,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestDepositionAutoConversion(StencilTest):
    PROGRAM = deposition_auto_conversion
    OUTPUTS = ("conversion_rate",)

    @static_reference
    def reference(grid, qi: np.ndarray, m_ice: np.ndarray, ice_dep: np.ndarray, **kwargs) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 6.6430804299795412e-08))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            qi=self.data_alloc.constant_field(
                2.02422e-2 + GraupelCt.qmin, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            m_ice=self.data_alloc.constant_field(1.0e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_dep=self.data_alloc.constant_field(
                2.06276e-05, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            conversion_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
