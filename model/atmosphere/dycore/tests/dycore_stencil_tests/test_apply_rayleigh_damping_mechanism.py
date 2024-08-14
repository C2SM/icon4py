# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import wpfloat


class TestApplyRayleighDampingMechanism(StencilTest):
    PROGRAM = apply_rayleigh_damping_mechanism
    OUTPUTS = ("w",)

    @staticmethod
    def reference(grid, z_raylfac: np.array, w_1: np.array, w: np.array, **kwargs) -> dict:
        z_raylfac = np.expand_dims(z_raylfac, axis=0)
        w_1 = np.expand_dims(w_1, axis=-1)
        w = z_raylfac * w + (1.0 - z_raylfac) * w_1
        return dict(w=w)

    @pytest.fixture
    def input_data(self, grid):
        z_raylfac = random_field(grid, KDim, dtype=wpfloat)
        w_1 = random_field(grid, CellDim, dtype=wpfloat)
        w = random_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            z_raylfac=z_raylfac,
            w_1=w_1,
            w=w,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
