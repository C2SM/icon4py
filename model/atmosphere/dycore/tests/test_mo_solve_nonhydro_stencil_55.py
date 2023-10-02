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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_55 import (
    mo_solve_nonhydro_stencil_55,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil55(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_55
    OUTPUTS = ("rho_new", "exner_new", "theta_v_new")

    @staticmethod
    def reference(
        mesh,
        z_rho_expl: np.array,
        vwind_impl_wgt: np.array,
        inv_ddqz_z_full: np.array,
        rho_ic: np.array,
        w: np.array,
        z_exner_expl: np.array,
        exner_ref_mc: np.array,
        z_alpha: np.array,
        z_beta: np.array,
        rho_now: np.array,
        theta_v_now: np.array,
        exner_now: np.array,
        dtime,
        cvd_o_rd,
        **kwargs,
    ) -> dict:
        rho_ic_offset_1 = rho_ic[:, 1:]
        w_offset_0 = w[:, :-1]
        w_offset_1 = w[:, 1:]
        z_alpha_offset_1 = z_alpha[:, 1:]
        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=1)
        rho_new = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full * (
            rho_ic[:, :-1] * w_offset_0 - rho_ic_offset_1 * w_offset_1
        )
        exner_new = (
            z_exner_expl
            + exner_ref_mc
            - z_beta * (z_alpha[:, :-1] * w_offset_0 - z_alpha_offset_1 * w_offset_1)
        )
        theta_v_new = (
            rho_now * theta_v_now * ((exner_new / exner_now - 1.0) * cvd_o_rd + 1.0) / rho_new
        )
        return dict(rho_new=rho_new, exner_new=exner_new, theta_v_new=theta_v_new)

    @pytest.fixture
    def input_data(self, mesh):
        z_rho_expl = random_field(mesh, CellDim, KDim)
        vwind_impl_wgt = random_field(mesh, CellDim)
        inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
        rho_ic = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        w = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        z_exner_expl = random_field(mesh, CellDim, KDim)
        exner_ref_mc = random_field(mesh, CellDim, KDim)
        z_alpha = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        z_beta = random_field(mesh, CellDim, KDim)
        rho_now = random_field(mesh, CellDim, KDim)
        theta_v_now = random_field(mesh, CellDim, KDim)
        exner_now = random_field(mesh, CellDim, KDim)
        rho_new = zero_field(mesh, CellDim, KDim)
        exner_new = zero_field(mesh, CellDim, KDim)
        theta_v_new = zero_field(mesh, CellDim, KDim)
        dtime = 5.0
        cvd_o_rd = 9.0

        return dict(
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=rho_now,
            theta_v_now=theta_v_now,
            exner_now=exner_now,
            rho_new=rho_new,
            exner_new=exner_new,
            theta_v_new=theta_v_new,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
        )
