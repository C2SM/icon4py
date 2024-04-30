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

from icon4py.model.common.constants import CPD_O_RD, GRAV_O_RD, P0REF
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure,
)
from icon4py.model.common.dimension import CellDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestDiagnoseSurfacePressure(StencilTest):
    PROGRAM = diagnose_surface_pressure
    OUTPUTS = ("pressure_sfc",)

    @staticmethod
    def reference(
        grid,
        exner_nlev_minus2: np.array,
        temperature_nlev: np.array,
        temperature_nlev_minus1: np.array,
        temperature_nlev_minus2: np.array,
        ddqz_z_full_nlev: np.array,
        ddqz_z_full_nlev_minus1: np.array,
        ddqz_z_full_nlev_minus2: np.array,
        **kwargs,
    ) -> dict:
        pressure_sfc = P0REF * np.exp(
            CPD_O_RD * np.log(exner_nlev_minus2)
            + GRAV_O_RD
            * (
                ddqz_z_full_nlev / temperature_nlev
                + ddqz_z_full_nlev_minus1 / temperature_nlev_minus1
                + 0.5 * ddqz_z_full_nlev_minus2 / temperature_nlev_minus2
            )
        )
        return dict(
            pressure_sfc=pressure_sfc,
        )

    @pytest.fixture
    def input_data(self, grid):
        pressure_sfc = zero_field(grid, CellDim, dtype=wpfloat)

        exner_nlev_minus2 = random_field(grid, CellDim, dtype=wpfloat)
        temperature_nlev = random_field(grid, CellDim, dtype=wpfloat)
        temperature_nlev_minus1 = random_field(grid, CellDim, dtype=wpfloat)
        temperature_nlev_minus2 = random_field(grid, CellDim, dtype=wpfloat)
        ddqz_z_full_nlev = random_field(grid, CellDim, dtype=wpfloat)
        ddqz_z_full_nlev_minus1 = random_field(grid, CellDim, dtype=wpfloat)
        ddqz_z_full_nlev_minus2 = random_field(grid, CellDim, dtype=wpfloat)

        return dict(
            exner_nlev_minus2=exner_nlev_minus2,
            temperature_nlev=temperature_nlev,
            temperature_nlev_minus1=temperature_nlev_minus1,
            temperature_nlev_minus2=temperature_nlev_minus2,
            ddqz_z_full_nlev=ddqz_z_full_nlev,
            ddqz_z_full_nlev_minus1=ddqz_z_full_nlev_minus1,
            ddqz_z_full_nlev_minus2=ddqz_z_full_nlev_minus2,
            pressure_sfc=pressure_sfc,
            cpd_o_rd=CPD_O_RD,
            p0ref=P0REF,
            grav_o_rd=GRAV_O_RD,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
        )
