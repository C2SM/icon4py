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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_43 import (
    mo_solve_nonhydro_stencil_43,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def mo_solve_nonhydro_stencil_43_numpy(
    grid,
    w_nnow: np.array,
    ddt_w_adv_ntl1: np.array,
    z_th_ddz_exner_c: np.array,
    rho_ic: np.array,
    w_concorr_c: np.array,
    vwind_expl_wgt: np.array,
    dtime: float,
    cpd: float,
) -> tuple[np.array, np.array]:
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
    z_w_expl = w_nnow + dtime * (ddt_w_adv_ntl1 - cpd * z_th_ddz_exner_c)
    z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
    return z_w_expl, z_contr_w_fl_l


class TestMoSolveNonhydroStencil43(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_43
    OUTPUTS = ("z_w_expl", "z_contr_w_fl_l")

    @staticmethod
    def reference(
        grid,
        w_nnow: np.array,
        ddt_w_adv_ntl1: np.array,
        z_th_ddz_exner_c: np.array,
        rho_ic: np.array,
        w_concorr_c: np.array,
        vwind_expl_wgt: np.array,
        dtime: float,
        cpd: float,
        **kwargs,
    ) -> dict:
        z_w_expl, z_contr_w_fl_l = mo_solve_nonhydro_stencil_43_numpy(
            grid,
            w_nnow,
            ddt_w_adv_ntl1,
            z_th_ddz_exner_c,
            rho_ic,
            w_concorr_c,
            vwind_expl_wgt,
            dtime,
            cpd,
        )
        return dict(z_w_expl=z_w_expl, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid):
        w_nnow = random_field(grid, CellDim, KDim)
        ddt_w_adv_ntl1 = random_field(grid, CellDim, KDim)
        z_th_ddz_exner_c = random_field(grid, CellDim, KDim)
        z_w_expl = zero_field(grid, CellDim, KDim)
        rho_ic = random_field(grid, CellDim, KDim)
        w_concorr_c = random_field(grid, CellDim, KDim)
        vwind_expl_wgt = random_field(grid, CellDim)
        z_contr_w_fl_l = zero_field(grid, CellDim, KDim)
        dtime = 5.0
        cpd = 10.0

        return dict(
            z_w_expl=z_w_expl,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            dtime=dtime,
            cpd=cpd,
        )
