# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.cloud_to_graupel import cloud_to_graupel
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestCloudToGraupelDefault(StencilTest):
    PROGRAM = cloud_to_graupel
    OUTPUTS = ("riming_graupel_rate",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qc: np.array, qg: np.array, TFRZ_HOM: wpfloat, QMIN: wpfloat, **kwargs) -> dict:
        return dict(riming_graupel_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t                   = constant_field(grid, 281.787, dims.CellDim, dtype=wpfloat),
            rho                 = constant_field(grid, 1.24783, dims.CellDim, dtype=wpfloat),
            qc                  = constant_field(grid, 0.0, dims.CellDim, dtype=wpfloat),
            qg                  = constant_field(grid, 1.03636e-25, dims.CellDim, dtype=wpfloat),
            TFRZ_HOM            = graupel_ct.tfrz_hom,
            QMIN                = graupel_ct.qmin,
            riming_graupel_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )

