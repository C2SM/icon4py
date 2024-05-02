# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import (
    diagnose_pressure,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestDiagnosePressure(StencilTest):
    PROGRAM = diagnose_pressure
    OUTPUTS = ("pressure_sfc",)

    @staticmethod
    def reference(
        grid,
        pressure_sfc: np.array,
        temperature: np.array,
        ddqz_z_full: np.array,
        **kwargs,
    ) -> dict:
        pressure_ifc = np.zeros_like(temperature)
        pressure = np.zeros_like(temperature)
        ground_level = temperature.shape[1] - 1
        pressure_ifc[:, ground_level] = pressure_sfc * np.exp(
            -ddqz_z_full[:, ground_level] / temperature[:, ground_level]
        )
        pressure[:, ground_level] = np.sqrt(pressure_ifc[:, ground_level] * pressure_sfc)
        for k in range(ground_level - 1, -1, -1):
            pressure_ifc[:, k] = pressure_ifc[:, k + 1] * np.exp(
                -ddqz_z_full[:, k] / temperature[:, k]
            )
            pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k + 1])

        return dict(
            pressure=pressure,
            pressure_ifc=pressure_ifc,
        )

    @pytest.fixture
    def input_data(self, grid):
        ddqz_z_full = random_field(grid, CellDim, KDim, dtype=wpfloat)
        temperature = random_field(grid, CellDim, KDim, dtype=vpfloat)
        pressure_sfc = random_field(grid, CellDim, dtype=vpfloat)
        pressure = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        pressure_ifc = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            ddqz_z_full=ddqz_z_full,
            temperature=temperature,
            pressure_sfc=pressure_sfc,
            pressure=pressure,
            pressure_ifc=pressure_ifc,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
