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

from icon4py.model.atmosphere.dycore.compute_solver_coefficients_matrix import (
    compute_solver_coefficients_matrix,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeSolverCoefficientsMatrix(StencilTest):
    PROGRAM = compute_solver_coefficients_matrix
    OUTPUTS = ("z_beta", "z_alpha")

    @staticmethod
    def reference(
        grid,
        exner_nnow: np.array,
        rho_nnow: np.array,
        theta_v_nnow: np.array,
        inv_ddqz_z_full: np.array,
        vwind_impl_wgt: np.array,
        theta_v_ic: np.array,
        rho_ic: np.array,
        dtime,
        rd,
        cvd,
        **kwargs,
    ) -> dict:
        z_beta = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full

        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
        z_alpha = vwind_impl_wgt * theta_v_ic * rho_ic
        return dict(z_beta=z_beta, z_alpha=z_alpha)

    @pytest.fixture
    def input_data(self, grid):
        exner_nnow = random_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_nnow = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_v_nnow = random_field(grid, CellDim, KDim, dtype=wpfloat)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim, dtype=vpfloat)
        vwind_impl_wgt = random_field(grid, CellDim, dtype=wpfloat)
        theta_v_ic = random_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_ic = random_field(grid, CellDim, KDim, dtype=wpfloat)
        z_alpha = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        z_beta = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        dtime = wpfloat("10.0")
        rd = wpfloat("5.0")
        cvd = wpfloat("3.0")

        return dict(
            z_beta=z_beta,
            exner_nnow=exner_nnow,
            rho_nnow=rho_nnow,
            theta_v_nnow=theta_v_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_alpha=z_alpha,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            rho_ic=rho_ic,
            dtime=dtime,
            rd=rd,
            cvd=cvd,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
