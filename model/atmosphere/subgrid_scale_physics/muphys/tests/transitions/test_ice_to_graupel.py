# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import ice_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestIceToGraupel(StencilTest):
    PROGRAM = ice_to_graupel
    OUTPUTS = ("aggregation",)

    @staticmethod
    def reference(grid, rho: np.array, qr: np.array, qg: np.array, qi: np.array, sticking_eff: np.array, **kwargs) -> dict:
        return dict(aggregation=np.full(rho.shape, 7.1049436957697864e-19))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            rho          = data_alloc.constant_field(grid, 1.04848, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr           = data_alloc.constant_field(grid, 6.00408e-13, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg           = data_alloc.constant_field(grid, 1.19022e-18, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi           = data_alloc.constant_field(grid, 1.9584e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            sticking_eff = data_alloc.constant_field(grid, 1.9584e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            aggregation  = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        )
