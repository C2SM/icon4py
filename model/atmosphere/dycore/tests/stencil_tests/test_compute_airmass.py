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
from icon4py.model.atmosphere.dycore.compute_airmass import compute_airmass
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import wpfloat


class TestComputeAirmass(StencilTest):
    PROGRAM = compute_airmass
    OUTPUTS = ("airmass_out",)

    @staticmethod
    def reference(
        grid, rho_in: np.array, ddqz_z_full_in: np.array, deepatmo_t1mc_in: np.array, **kwargs
    ) -> dict:
        airmass_out = rho_in * ddqz_z_full_in * deepatmo_t1mc_in
        return dict(airmass_out=airmass_out)

    @pytest.fixture
    def input_data(self, grid):
        rho_in = random_field(grid, CellDim, KDim, dtype=wpfloat)
        ddqz_z_full_in = random_field(grid, CellDim, KDim, dtype=wpfloat)
        deepatmo_t1mc_in = random_field(grid, KDim, dtype=wpfloat)
        airmass_out = random_field(grid, CellDim, KDim, dtype=wpfloat)
        return dict(
            rho_in=rho_in,
            ddqz_z_full_in=ddqz_z_full_in,
            deepatmo_t1mc_in=deepatmo_t1mc_in,
            airmass_out=airmass_out,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
