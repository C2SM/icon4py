# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import snow_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestSnowToGraupel(StencilTest):
    PROGRAM = snow_to_graupel
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(
        grid, t: np.ndarray, rho: np.ndarray, qc: np.ndarray, qs: np.ndarray, **kwargs
    ) -> dict:
        return dict(conversion_rate=np.full(t.shape, 6.2696154545048011e-10))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 265.85, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(grid, 1.04848, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.constant_field(grid, 7.02792e-5, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=data_alloc.constant_field(grid, 4.44664e-7, dims.CellDim, dims.KDim, dtype=wpfloat),
            conversion_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
