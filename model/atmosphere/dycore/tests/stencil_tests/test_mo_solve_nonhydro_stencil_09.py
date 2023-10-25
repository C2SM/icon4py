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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_09 import (
    mo_solve_nonhydro_stencil_09,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil09(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_09
    OUTPUTS = ("z_theta_v_pr_ic", "theta_v_ic", "z_th_ddz_exner_c")

    @staticmethod
    def reference(
        grid,
        wgtfac_c: np.array,
        z_rth_pr_2: np.array,
        theta_v: np.array,
        vwind_expl_wgt: np.array,
        exner_pr: np.array,
        d_exner_dz_ref_ic: np.array,
        ddqz_z_half: np.array,
        **kwargs,
    ) -> tuple[np.array, np.array, np.array]:
        z_rth_pr_2_offset = np.roll(z_rth_pr_2, axis=1, shift=1)
        theta_v_offset = np.roll(theta_v, axis=1, shift=1)
        exner_pr_offset = np.roll(exner_pr, axis=1, shift=1)
        vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)

        z_theta_v_pr_ic = wgtfac_c * z_rth_pr_2 + (1.0 - wgtfac_c) * z_rth_pr_2_offset
        z_theta_v_pr_ic[:, 0] = 0
        theta_v_ic = wgtfac_c * theta_v + (1 - wgtfac_c) * theta_v_offset
        theta_v_ic[:, 0] = 0
        z_th_ddz_exner_c = (
            vwind_expl_wgt * theta_v_ic * (exner_pr_offset - exner_pr) / ddqz_z_half
            + z_theta_v_pr_ic * d_exner_dz_ref_ic
        )
        z_th_ddz_exner_c[:, 0] = 0
        return dict(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        )

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_c = random_field(grid, CellDim, KDim)
        z_rth_pr_2 = random_field(grid, CellDim, KDim)
        theta_v = random_field(grid, CellDim, KDim)
        vwind_expl_wgt = random_field(grid, CellDim)
        exner_pr = random_field(grid, CellDim, KDim)
        d_exner_dz_ref_ic = random_field(grid, CellDim, KDim)
        ddqz_z_half = random_field(grid, CellDim, KDim)
        z_theta_v_pr_ic = zero_field(grid, CellDim, KDim)
        theta_v_ic = zero_field(grid, CellDim, KDim)
        z_th_ddz_exner_c = zero_field(grid, CellDim, KDim)

        return dict(
            wgtfac_c=wgtfac_c,
            z_rth_pr_2=z_rth_pr_2,
            theta_v=theta_v,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        )
