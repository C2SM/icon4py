# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import T_from_internal_energy

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestTFromInternalEnergy(StencilTest):
    PROGRAM = T_from_internal_energy
    OUTPUTS = ("temperature",)

    @staticmethod
    def reference(grid, u: np.array, qv: np.array, qliq: np.array, qice: np.array, rho: np.array, dz: np.array, **kwargs) -> dict:
        return dict(temperature=np.full(u.shape, 255.75599999999997))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            u                = constant_field(grid, 38265357.270336017, dims.CellDim, dims.KDim, dtype=wpfloat),
            qv               = constant_field(grid, 0.00122576, dims.CellDim, dims.KDim, dtype=wpfloat),
            qliq             = constant_field(grid, 1.63837e-20, dims.CellDim, dims.KDim, dtype=wpfloat),
            qice             = constant_field(grid, 1.09462e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho              = constant_field(grid, 0.83444, dims.CellDim, dims.KDim, dtype=wpfloat),
            dz               = constant_field(grid, 249.569, dims.CellDim, dims.KDim, dtype=wpfloat),
            temperature      = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )
