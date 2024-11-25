# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.muphys.core.transitions import cloud_to_snow
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestUpdateWind(StencilTest):
    PROGRAM = cloud_to_snow
    OUTPUTS = ("riming_snow_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qs: np.array, ns: np.array, lam: np.array, **kwargs) -> dict:
        return dict(riming_snow_rate=np.full(t.shape, 9.5431874564438999e-10))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t       = constant_field(grid, 256.571, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc      = constant_field(grid, 3.31476e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs      = constant_field(grid, 7.47365e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns      = constant_field(grid, 3.37707e+07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam     = constant_field(grid, 8989.78, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

