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

from icon4py.model.atm_dyn_iconam.mo_solve_nonhydro_stencil_42 import (
    mo_solve_nonhydro_stencil_42,
)
from icon4py.model.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_42_numpy(
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
) -> tuple[np.array]:
    z_w_expl = w_nnow + dtime * (
        wgt_nnow_vel * ddt_w_adv_ntl1
        + wgt_nnew_vel * ddt_w_adv_ntl2
        - cpd * z_th_ddz_exner_c
    )
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
    z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
    return z_w_expl, z_contr_w_fl_l


def test_mo_solve_nonhydro_stencil_42():
    mesh = SimpleMesh()

    w_nnow = random_field(mesh, CellDim, KDim)
    ddt_w_adv_ntl1 = random_field(mesh, CellDim, KDim)
    ddt_w_adv_ntl2 = random_field(mesh, CellDim, KDim)
    z_th_ddz_exner_c = random_field(mesh, CellDim, KDim)
    z_w_expl = zero_field(mesh, CellDim, KDim)
    rho_ic = random_field(mesh, CellDim, KDim)
    w_concorr_c = random_field(mesh, CellDim, KDim)
    vwind_expl_wgt = random_field(mesh, CellDim)
    z_contr_w_fl_l = zero_field(mesh, CellDim, KDim)
    dtime = 5.0
    wgt_nnow_vel = 8.0
    wgt_nnew_vel = 9.0
    cpd = 10.0

    z_w_expl_ref, z_contr_w_fl_l_ref = mo_solve_nonhydro_stencil_42_numpy(
        np.asarray(w_nnow),
        np.asarray(ddt_w_adv_ntl1),
        np.asarray(ddt_w_adv_ntl2),
        np.asarray(z_th_ddz_exner_c),
        np.asarray(rho_ic),
        np.asarray(w_concorr_c),
        np.asarray(vwind_expl_wgt),
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
    )
    mo_solve_nonhydro_stencil_42(
        z_w_expl,
        w_nnow,
        ddt_w_adv_ntl1,
        ddt_w_adv_ntl2,
        z_th_ddz_exner_c,
        z_contr_w_fl_l,
        rho_ic,
        w_concorr_c,
        vwind_expl_wgt,
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
        offset_provider={},
    )

    assert np.allclose(z_w_expl, z_w_expl_ref)
    assert np.allclose(z_contr_w_fl_l, z_contr_w_fl_l_ref)
