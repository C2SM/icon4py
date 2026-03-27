# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import deposition_factor
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestDepositionFactor(StencilTest):
    PROGRAM = deposition_factor
    OUTPUTS = ("deposition_rate",)

    @static_reference
    def reference(grid, t: np.ndarray, qvsi: np.ndarray, **kwargs) -> dict:
        return dict(deposition_rate=np.full(t.shape, 1.3234329478493952e-05))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            t=self.data_alloc.constant_field(272.731, dims.CellDim, dims.KDim, dtype=wpfloat),
            qvsi=self.data_alloc.constant_field(
                0.00416891, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            deposition_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
