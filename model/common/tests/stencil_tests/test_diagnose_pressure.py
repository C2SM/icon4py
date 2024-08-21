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

from icon4py.model.common import constants as phy_const, dimension as dims, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import (
    diagnose_pressure,
)
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    is_roundtrip,
    random_field,
    zero_field,
)


class TestDiagnosePressure(StencilTest):
    PROGRAM = diagnose_pressure
    OUTPUTS = ("pressure", "pressure_ifc")

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
            -phy_const.GRAV_O_RD * ddqz_z_full[:, ground_level] / temperature[:, ground_level]
        )
        pressure[:, ground_level] = np.sqrt(pressure_ifc[:, ground_level] * pressure_sfc)
        for k in range(ground_level - 1, -1, -1):
            pressure_ifc[:, k] = pressure_ifc[:, k + 1] * np.exp(
                -phy_const.GRAV_O_RD * ddqz_z_full[:, k] / temperature[:, k]
            )
            pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k + 1])

        return dict(
            pressure=pressure,
            pressure_ifc=pressure_ifc,
        )

    @pytest.fixture
    def input_data(self, grid):
        if is_roundtrip:
            pytest.xfail("This stencil currently does not work properly with roundtrip backend.")

        ddqz_z_full = random_field(grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat)
        virtual_temperature = random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat
        )
        pressure_sfc = random_field(grid, dims.CellDim, low=1.0e-6, dtype=ta.wpfloat)
        pressure = zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        pressure_ifc = zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            ddqz_z_full=ddqz_z_full,
            virtual_temperature=virtual_temperature,
            pressure_sfc=pressure_sfc,
            pressure=pressure,
            pressure_ifc=pressure_ifc,
            grav_o_rd=phy_const.GRAV_O_RD,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
