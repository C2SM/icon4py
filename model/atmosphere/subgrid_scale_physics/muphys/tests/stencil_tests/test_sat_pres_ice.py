# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import sat_pres_ice
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestSatPresIce(StencilTest):
    PROGRAM = sat_pres_ice
    OUTPUTS = ("pressure",)

    @staticmethod
    def reference(grid, t: np.ndarray, **kwargs) -> dict:
        return dict(pressure=np.full(t.shape, 1216.7746246067475))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 281.787, dims.CellDim, dims.KDim, dtype=wpfloat),
            pressure=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
