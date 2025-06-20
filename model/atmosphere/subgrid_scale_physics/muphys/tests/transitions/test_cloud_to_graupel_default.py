# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestCloudToGraupelDefault(StencilTest):
    PROGRAM = cloud_to_graupel
    OUTPUTS = ("riming_graupel_rate",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qc: np.array, qg: np.array, **kwargs) -> dict:
        return dict(riming_graupel_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t                   = data_alloc.constant_field(grid, 281.787, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho                 = data_alloc.constant_field(grid, 1.24783, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc                  = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg                  = data_alloc.constant_field(grid, 1.03636e-25, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_graupel_rate = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        )

