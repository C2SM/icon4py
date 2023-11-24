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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_50 import (
    mo_solve_nonhydro_stencil_50,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


def mo_solve_nonhydro_stencil_50_numpy(
    grid,
    z_rho_expl: np.array,
    rho_incr: np.array,
    z_exner_expl: np.array,
    exner_incr: np.array,
    iau_wgt_dyn: float,
) -> tuple[np.array, np.array]:
    z_rho_expl = z_rho_expl + iau_wgt_dyn * rho_incr
    z_exner_expl = z_exner_expl + iau_wgt_dyn * exner_incr
    return z_rho_expl, z_exner_expl


class TestMoSolveNonhydroStencil50(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_50
    OUTPUTS = ("z_rho_expl", "z_exner_expl")

    @staticmethod
    def reference(
        grid,
        z_rho_expl: np.array,
        rho_incr: np.array,
        z_exner_expl: np.array,
        exner_incr: np.array,
        iau_wgt_dyn: float,
        **kwargs,
    ) -> dict:
        z_rho_expl, z_exner_expl = mo_solve_nonhydro_stencil_50_numpy(
            grid,
            z_rho_expl,
            rho_incr,
            z_exner_expl,
            exner_incr,
            iau_wgt_dyn,
        )
        return dict(z_rho_expl=z_rho_expl, z_exner_expl=z_exner_expl)

    @pytest.fixture
    def input_data(self, grid):
        z_exner_expl = random_field(grid, CellDim, KDim)
        exner_incr = random_field(grid, CellDim, KDim)
        z_rho_expl = random_field(grid, CellDim, KDim)
        rho_incr = random_field(grid, CellDim, KDim)
        iau_wgt_dyn = 8.0

        return dict(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
