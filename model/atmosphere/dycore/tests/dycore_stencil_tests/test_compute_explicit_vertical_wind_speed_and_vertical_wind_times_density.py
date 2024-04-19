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

from icon4py.model.atmosphere.dycore.compute_explicit_vertical_wind_speed_and_vertical_wind_times_density import (
    compute_explicit_vertical_wind_speed_and_vertical_wind_times_density,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeExplicitVerticalWindSpeedAndVerticalWindTimesDensity(StencilTest):
    PROGRAM = compute_explicit_vertical_wind_speed_and_vertical_wind_times_density
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
        vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, -1)
        z_w_expl = w_nnow + dtime * (ddt_w_adv_ntl1 - cpd * z_th_ddz_exner_c)
        z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
        return dict(z_w_expl=z_w_expl, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid):
        w_nnow = random_field(grid, CellDim, KDim, dtype=wpfloat)
        ddt_w_adv_ntl1 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_th_ddz_exner_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_w_expl = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_ic = random_field(grid, CellDim, KDim, dtype=wpfloat)
        w_concorr_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        vwind_expl_wgt = random_field(grid, CellDim, dtype=wpfloat)
        z_contr_w_fl_l = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        cpd = wpfloat("10.0")

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
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
