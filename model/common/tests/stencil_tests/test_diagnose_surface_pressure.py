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

from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import CPD_O_RD, GRAV_O_RD, P0REF
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure,
)
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestDiagnoseSurfacePressure(StencilTest):
    PROGRAM = diagnose_surface_pressure
    OUTPUTS = ("pressure_sfc",)

    @staticmethod
    def reference(
        grid,
        exner: np.array,
        temperature: np.array,
        ddqz_z_full: np.array,
        **kwargs,
    ) -> dict:
        pressure_sfc = np.zeros((grid.num_cells, grid.num_levels + 1), dtype=vpfloat)
        pressure_sfc[:, -1] = P0REF * np.exp(
            CPD_O_RD * np.log(exner[:, -3])
            + GRAV_O_RD
            * (
                ddqz_z_full[:, -1] / temperature[:, -1]
                + ddqz_z_full[:, -2] / temperature[:, -2]
                + 0.5 * ddqz_z_full[:, -3] / temperature[:, -3]
            )
        )
        return dict(
            pressure_sfc=pressure_sfc,
        )

    @pytest.fixture
    def input_data(self, grid):
        exner = random_field(grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=vpfloat)
        temperature = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddqz_z_full = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        pressure_sfc = zero_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat, extend={dims.KDim: 1}
        )

        return dict(
            exner=exner,
            temperature=temperature,
            ddqz_z_full=ddqz_z_full,
            pressure_sfc=pressure_sfc,
            cpd_o_rd=CPD_O_RD,
            p0ref=P0REF,
            grav_o_rd=GRAV_O_RD,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(grid.num_levels),
            vertical_end=int32(grid.num_levels + 1),
        )
