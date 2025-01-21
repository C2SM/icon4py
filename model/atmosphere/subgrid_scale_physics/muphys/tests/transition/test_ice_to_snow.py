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


class TestIceToSnow(StencilTest):
    PROGRAM = ice_to_snow
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, qi: np.array, ns: np.array, lam: np.array, sticking_eff: np.array, QMIN: wpfloat, V0S: wpfloat, V1S: wpfloat,  **kwargs) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 3.3262745200740486e-11))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qi              = constant_field(grid, 6.43223e-08, dims.CellDim, dtype=wpfloat),
            ns              = constant_field(grid, 1.93157e+07, dims.CellDim, dtype=wpfloat),
            lam             = constant_field(grid, 10576.8, dims.CellDim, dtype=wpfloat),
            sticking_eff    = constant_field(grid, 0.511825, dims.CellDim, dtype=wpfloat),
            QMIN            = graupel_ct.qmin,
            V0S             = graupel_ct.v0s,
            V1S             = graupel_ct.v1s,
            conversion_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
