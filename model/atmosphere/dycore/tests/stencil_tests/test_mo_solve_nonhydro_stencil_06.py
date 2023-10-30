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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_06 import (
    mo_solve_nonhydro_stencil_06,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field

def mo_solve_nonhydro_stencil_06_numpy(
    mesh,
    inv_ddqz_z_full: np.array,
    z_exner_ic: np.array,
    z_exner_ex_pr_offset_1: np.array,
) -> np.array:
    z_dexner_dz_c_1 = (z_exner_ic[:, :-1] - z_exner_ic[:, 1:]) * inv_ddqz_z_full
    return z_dexner_dz_c_1

class TestMoSolveNonhydroStencil06(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_06
    OUTPUTS = ("z_dexner_dz_c_1",)

    @staticmethod
    def reference(mesh, z_exner_ic: np.array, inv_ddqz_z_full: np.array, **kwargs) -> np.array:
        z_dexner_dz_c_1 = mo_solve_nonhydro_stencil_06_numpy(z_exner_ic,
                                                             inv_ddqz_z_full)
        return dict(z_dexner_dz_c_1=z_dexner_dz_c_1)

    @pytest.fixture
    def input_data(self, mesh):
        z_exner_ic = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
        z_dexner_dz_c_1 = zero_field(mesh, CellDim, KDim)

        return dict(
            z_exner_ic=z_exner_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
        )
