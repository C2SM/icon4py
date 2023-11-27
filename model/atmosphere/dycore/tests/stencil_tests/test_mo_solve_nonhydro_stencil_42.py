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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_42 import (
    mo_solve_nonhydro_stencil_42,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil42(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_42
    OUTPUTS = ("z_w_expl", "z_contr_w_fl_l")

    @staticmethod
    def reference(
        grid,
        w_nnow: np.array,
        ddt_w_adv_ntl1: np.array,
        ddt_w_adv_ntl2: np.array,
        z_th_ddz_exner_c: np.array,
        rho_ic: np.array,
        w_concorr_c: np.array,
        vwind_expl_wgt: np.array,
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
        **kwargs,
    ) -> tuple[np.array]:
        z_w_expl = w_nnow + dtime * (
            wgt_nnow_vel * ddt_w_adv_ntl1 + wgt_nnew_vel * ddt_w_adv_ntl2 - cpd * z_th_ddz_exner_c
        )
        vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
        z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
        return dict(z_w_expl=z_w_expl, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid):
        w_nnow = random_field(grid, CellDim, KDim, dtype=wpfloat)
        ddt_w_adv_ntl1 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        ddt_w_adv_ntl2 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_th_ddz_exner_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_w_expl = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_ic = random_field(grid, CellDim, KDim, dtype=wpfloat)
        w_concorr_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        vwind_expl_wgt = random_field(grid, CellDim, dtype=wpfloat)
        z_contr_w_fl_l = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        wgt_nnow_vel = wpfloat("8.0")
        wgt_nnew_vel = wpfloat("9.0")
        cpd = wpfloat("10.0")

        return dict(
            z_w_expl=z_w_expl,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
        )
