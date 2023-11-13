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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_61 import (
    mo_solve_nonhydro_stencil_61,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil61(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_61
    OUTPUTS = ("rho_new", "exner_new", "w_new")

    @staticmethod
    def reference(
        grid,
        rho_now: np.array,
        grf_tend_rho: np.array,
        theta_v_now: np.array,
        grf_tend_thv: np.array,
        w_now: np.array,
        grf_tend_w: np.array,
        dtime,
        **kwargs,
    ) -> tuple[np.array]:
        rho_new = rho_now + dtime * grf_tend_rho
        exner_new = theta_v_now + dtime * grf_tend_thv
        w_new = w_now + dtime * grf_tend_w
        return dict(rho_new=rho_new, exner_new=exner_new, w_new=w_new)

    @pytest.fixture
    def input_data(self, grid):
        rho_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        grf_tend_rho = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_v_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        grf_tend_thv = random_field(grid, CellDim, KDim, dtype=wpfloat)
        w_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        grf_tend_w = random_field(grid, CellDim, KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        rho_new = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        exner_new = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        w_new = zero_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            rho_now=rho_now,
            grf_tend_rho=grf_tend_rho,
            theta_v_now=theta_v_now,
            grf_tend_thv=grf_tend_thv,
            w_now=w_now,
            grf_tend_w=grf_tend_w,
            rho_new=rho_new,
            exner_new=exner_new,
            w_new=w_new,
            dtime=dtime,
        )
