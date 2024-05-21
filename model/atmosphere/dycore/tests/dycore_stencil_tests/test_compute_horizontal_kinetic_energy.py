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

from icon4py.model.atmosphere.dycore.compute_horizontal_kinetic_energy import (
    compute_horizontal_kinetic_energy,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_horizontal_kinetic_energy_numpy(vn: np.array, vt: np.array) -> tuple:
    vn_ie = vn
    z_vt_ie = vt
    z_kin_hor_e = 0.5 * ((vn * vn) + (vt * vt))
    return vn_ie, z_vt_ie, z_kin_hor_e


class TestComputeHorizontalKineticEnergy(StencilTest):
    PROGRAM = compute_horizontal_kinetic_energy
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @staticmethod
    def reference(grid, vn: np.array, vt: np.array, **kwargs) -> dict:
        vn_ie, z_vt_ie, z_kin_hor_e = compute_horizontal_kinetic_energy_numpy(vn, vt)
        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)

        vn_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_vt_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_kin_hor_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
