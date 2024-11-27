# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.ice_to_snow import ice_to_snow
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestIceToSnowDefault(StencilTest):
    PROGRAM = ice_to_snow
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, qi: np.array, ns: np.array, lam: np.array, sticking_eff: np.array, QMIN: wpfloat, V0S: wpfloat, V1S: wpfloat,  **kwargs) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qi              = constant_field(grid, 7.95122e-25, dims.CellDim, dtype=wpfloat),
            ns              = constant_field(grid, 2.23336e+07, dims.CellDim, dtype=wpfloat),
            lam             = constant_field(grid, 61911.1, dims.CellDim, dtype=wpfloat),
            sticking_eff    = constant_field(grid, 0.241568, dims.CellDim, dtype=wpfloat),
            QMIN            = graupel_ct.qmin,
            V0S             = graupel_ct.v0s,
            V1S             = graupel_ct.v1s,
            conversion_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
