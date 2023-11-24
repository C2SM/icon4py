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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_48 import (
    mo_solve_nonhydro_stencil_48,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil48(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_48
    OUTPUTS = ("z_rho_expl", "z_exner_expl")

    @staticmethod
    def reference(
        grid,
        rho_nnow: np.array,
        inv_ddqz_z_full: np.array,
        z_flxdiv_mass: np.array,
        z_contr_w_fl_l: np.array,
        exner_pr: np.array,
        z_beta: np.array,
        z_flxdiv_theta: np.array,
        theta_v_ic: np.array,
        ddt_exner_phy: np.array,
        dtime,
        **kwargs,
    ) -> dict:
        z_rho_expl = rho_nnow - dtime * inv_ddqz_z_full * (
            z_flxdiv_mass + z_contr_w_fl_l[:, :-1] - z_contr_w_fl_l[:, 1:]
        )

        z_exner_expl = (
            exner_pr
            - z_beta
            * (
                z_flxdiv_theta
                + (theta_v_ic * z_contr_w_fl_l)[:, :-1]
                - (theta_v_ic * z_contr_w_fl_l)[:, 1:]
            )
            + dtime * ddt_exner_phy
        )
        return dict(z_rho_expl=z_rho_expl, z_exner_expl=z_exner_expl)

    @pytest.fixture
    def input_data(self, grid):
        dtime = 1.0
        rho_nnow = random_field(grid, CellDim, KDim)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim)
        z_flxdiv_mass = random_field(grid, CellDim, KDim)
        z_contr_w_fl_l = random_field(grid, CellDim, KDim, extend={KDim: 1})
        exner_pr = random_field(grid, CellDim, KDim)
        z_beta = random_field(grid, CellDim, KDim)
        z_flxdiv_theta = random_field(grid, CellDim, KDim)
        theta_v_ic = random_field(grid, CellDim, KDim, extend={KDim: 1})
        ddt_exner_phy = random_field(grid, CellDim, KDim)

        z_rho_expl = zero_field(grid, CellDim, KDim)
        z_exner_expl = zero_field(grid, CellDim, KDim)

        return dict(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_nnow=rho_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_flxdiv_mass=z_flxdiv_mass,
            z_contr_w_fl_l=z_contr_w_fl_l,
            exner_pr=exner_pr,
            z_beta=z_beta,
            z_flxdiv_theta=z_flxdiv_theta,
            theta_v_ic=theta_v_ic,
            ddt_exner_phy=ddt_exner_phy,
            dtime=dtime,
        )
