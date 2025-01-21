# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.ice_to_graupel import ice_to_graupel
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestIceToGraupel(StencilTest):
    PROGRAM = ice_to_graupel
    OUTPUTS = ("aggregation",)

    @staticmethod
    def reference(grid, rho: np.array, qr: np.array, qg: np.array, qi: np.array, sticking_eff: np.array, QMIN: wpfloat, **kwargs) -> dict:
        return dict(aggregation=np.full(rho.shape, 7.1049436957697864e-19))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            rho          = constant_field(grid, 1.04848, dims.CellDim, dtype=wpfloat),
            qr           = constant_field(grid, 6.00408e-13, dims.CellDim, dtype=wpfloat),
            qg           = constant_field(grid, 1.19022e-18, dims.CellDim, dtype=wpfloat),
            qi           = constant_field(grid, 1.9584e-08, dims.CellDim, dtype=wpfloat),
            sticking_eff = constant_field(grid, 1.9584e-08, dims.CellDim, dtype=wpfloat),
            QMIN         = graupel_ct.qmin,
            aggregation  = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
