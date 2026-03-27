# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    vel_scale_factor_ice,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestVelScaleFactorIce(StencilTest):
    PROGRAM = vel_scale_factor_ice
    OUTPUTS = ("scale_factor",)

    @static_reference
    def reference(grid, xrho: np.ndarray, **kwargs) -> dict:
        return dict(scale_factor=np.full(xrho.shape, 1.1158596098981044))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            xrho=self.data_alloc.constant_field(1.17873, dims.CellDim, dims.KDim, dtype=wpfloat),
            scale_factor=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
