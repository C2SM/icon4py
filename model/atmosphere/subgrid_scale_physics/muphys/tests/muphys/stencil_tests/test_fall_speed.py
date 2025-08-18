# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import Idx
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    fall_speed,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestFallSpeed(StencilTest):
    PROGRAM = fall_speed
    OUTPUTS = ("speed",)

    @staticmethod
    def reference(
        grid, density: np.ndarray, prefactor: wpfloat, offset: wpfloat, exponent: wpfloat, **kwargs
    ) -> dict:
        return dict(speed=np.full(density.shape, 0.67882452435647411))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            density=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            prefactor=Idx.prefactor_r,
            offset=Idx.offset_r,
            exponent=Idx.exponent_r,
            speed=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
