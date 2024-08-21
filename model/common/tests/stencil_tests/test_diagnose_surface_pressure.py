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
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure,
)
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestDiagnoseSurfacePressure(StencilTest):
    PROGRAM = diagnose_surface_pressure
    OUTPUTS = ("pressure_sfc",)

    @staticmethod
    def reference(
        grid,
        exner: np.array,
        virtual_temperature: np.array,
        ddqz_z_full: np.array,
        **kwargs,
    ) -> dict:
        pressure_sfc = np.zeros((grid.num_cells, grid.num_levels + 1), dtype=ta.wpfloat)
        pressure_sfc[:, -1] = phy_const.P0REF * np.exp(
            phy_const.CPD_O_RD * np.log(exner[:, -3])
            + phy_const.GRAV_O_RD
            * (
                ddqz_z_full[:, -1] / virtual_temperature[:, -1]
                + ddqz_z_full[:, -2] / virtual_temperature[:, -2]
                + 0.5 * ddqz_z_full[:, -3] / virtual_temperature[:, -3]
            )
        )
        return dict(
            pressure_sfc=pressure_sfc,
        )

    @pytest.fixture
    def input_data(self, grid):
        exner = random_field(grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat)
        virtual_temperature = random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat
        )
        ddqz_z_full = random_field(grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat)
        pressure_sfc = zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            exner=exner,
            virtual_temperature=virtual_temperature,
            ddqz_z_full=ddqz_z_full,
            pressure_sfc=pressure_sfc,
            cpd_o_rd=phy_const.CPD_O_RD,
            p0ref=phy_const.P0REF,
            grav_o_rd=phy_const.GRAV_O_RD,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(grid.num_levels),
            vertical_end=int32(grid.num_levels + 1),
        )
