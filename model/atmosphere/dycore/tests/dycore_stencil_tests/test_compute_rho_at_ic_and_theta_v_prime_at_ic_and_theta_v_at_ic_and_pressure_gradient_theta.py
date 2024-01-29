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

from icon4py.model.atmosphere.dycore.compute_rho_at_ic_and_theta_v_prime_at_ic_and_theta_v_at_ic_and_pressure_gradient_theta import (
    compute_rho_at_ic_and_theta_v_prime_at_ic_and_theta_v_at_ic_and_pressure_gradient_theta,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil10(StencilTest):
    PROGRAM = compute_rho_at_ic_and_theta_v_prime_at_ic_and_theta_v_at_ic_and_pressure_gradient_theta
    OUTPUTS = ("rho_ic", "z_theta_v_pr_ic", "theta_v_ic", "z_th_ddz_exner_c")

    @staticmethod
    def reference(
        grid,
        w: np.array,
        w_concorr_c: np.array,
        ddqz_z_half: np.array,
        rho_now: np.array,
        rho_var: np.array,
        theta_now: np.array,
        theta_var: np.array,
        wgtfac_c: np.array,
        theta_ref_mc: np.array,
        vwind_expl_wgt: np.array,
        exner_pr: np.array,
        d_exner_dz_ref_ic: np.array,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        **kwargs,
    ) -> tuple[np.array, np.array, np.array, np.array]:
        vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
        rho_now_offset = np.roll(rho_now, shift=1, axis=1)
        rho_var_offset = np.roll(rho_var, shift=1, axis=1)
        theta_now_offset = np.roll(theta_now, shift=1, axis=1)
        theta_var_offset = np.roll(theta_var, shift=1, axis=1)
        theta_ref_mc_offset = np.roll(theta_ref_mc, shift=1, axis=1)
        exner_pr_offset = np.roll(exner_pr, shift=1, axis=1)

        z_w_backtraj = -(w - w_concorr_c) * dtime * 0.5 / ddqz_z_half
        z_rho_tavg_m1 = wgt_nnow_rth * rho_now_offset + wgt_nnew_rth * rho_var_offset
        z_theta_tavg_m1 = wgt_nnow_rth * theta_now_offset + wgt_nnew_rth * theta_var_offset
        z_rho_tavg = wgt_nnow_rth * rho_now + wgt_nnew_rth * rho_var
        z_theta_tavg = wgt_nnow_rth * theta_now + wgt_nnew_rth * theta_var
        rho_ic = (
            wgtfac_c * z_rho_tavg
            + (1 - wgtfac_c) * z_rho_tavg_m1
            + z_w_backtraj * (z_rho_tavg_m1 - z_rho_tavg)
        )
        rho_ic[:, 0] = 0
        z_theta_v_pr_mc_m1 = z_theta_tavg_m1 - theta_ref_mc_offset
        z_theta_v_pr_mc = z_theta_tavg - theta_ref_mc
        z_theta_v_pr_ic = wgtfac_c * z_theta_v_pr_mc + (1 - wgtfac_c) * z_theta_v_pr_mc_m1
        z_theta_v_pr_ic[:, 0] = 0
        theta_v_ic = (
            wgtfac_c * z_theta_tavg
            + (1 - wgtfac_c) * z_theta_tavg_m1
            + z_w_backtraj * (z_theta_tavg_m1 - z_theta_tavg)
        )
        theta_v_ic[:, 0] = 0
        z_th_ddz_exner_c = (
            vwind_expl_wgt * theta_v_ic * (exner_pr_offset - exner_pr) / ddqz_z_half
            + z_theta_v_pr_ic * d_exner_dz_ref_ic
        )
        z_th_ddz_exner_c[:, 0] = 0
        return dict(
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        )

    @pytest.fixture
    def input_data(self, grid):
        dtime = wpfloat("1.0")
        wgt_nnow_rth = wpfloat("2.0")
        wgt_nnew_rth = wpfloat("3.0")
        w = random_field(grid, CellDim, KDim, dtype=wpfloat)
        w_concorr_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        ddqz_z_half = random_field(grid, CellDim, KDim, dtype=vpfloat)
        rho_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_var = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_var = random_field(grid, CellDim, KDim, dtype=wpfloat)
        wgtfac_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        theta_ref_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)
        vwind_expl_wgt = random_field(grid, CellDim, dtype=wpfloat)
        exner_pr = random_field(grid, CellDim, KDim, dtype=wpfloat)
        d_exner_dz_ref_ic = random_field(grid, CellDim, KDim, dtype=vpfloat)
        rho_ic = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        z_theta_v_pr_ic = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        theta_v_ic = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        z_th_ddz_exner_c = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        return dict(
            w=w,
            w_concorr_c=w_concorr_c,
            ddqz_z_half=ddqz_z_half,
            rho_now=rho_now,
            rho_var=rho_var,
            theta_now=theta_now,
            theta_var=theta_var,
            wgtfac_c=wgtfac_c,
            theta_ref_mc=theta_ref_mc,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
