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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_48 import (
    mo_solve_nonhydro_stencil_48,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_48_numpy(
    dtime: float,
    rho_nnow: np.array,
    inv_ddqz_z_full: np.array,
    z_flxdiv_mass: np.array,
    z_contr_w_fl_l: np.array,
    exner_pr: np.array,
    z_beta: np.array,
    z_flxdiv_theta: np.array,
    theta_v_ic: np.array,
    ddt_exner_phy: np.array,
) -> tuple[np.array, np.array]:
    z_contr_w_fl_l_offset_1 = np.roll(z_contr_w_fl_l, shift=-1, axis=1)
    theta_v_ic_offset_1 = np.roll(theta_v_ic, shift=-1, axis=1)

    z_rho_expl = rho_nnow - dtime * inv_ddqz_z_full * (
        z_flxdiv_mass + z_contr_w_fl_l - z_contr_w_fl_l_offset_1
    )

    z_exner_expl = (
        exner_pr
        - z_beta
        * (
            z_flxdiv_theta
            + theta_v_ic * z_contr_w_fl_l
            - theta_v_ic_offset_1 * z_contr_w_fl_l_offset_1
        )
        + dtime * ddt_exner_phy
    )
    return z_rho_expl, z_exner_expl


def test_mo_solve_nonhydro_stencil_48():
    mesh = SimpleMesh()

    dtime = 1.0
    rho_nnow = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    z_flxdiv_mass = random_field(mesh, CellDim, KDim)
    z_contr_w_fl_l = random_field(mesh, CellDim, KDim)
    exner_pr = random_field(mesh, CellDim, KDim)
    z_beta = random_field(mesh, CellDim, KDim)
    z_flxdiv_theta = random_field(mesh, CellDim, KDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    ddt_exner_phy = random_field(mesh, CellDim, KDim)

    z_rho_expl = zero_field(mesh, CellDim, KDim)
    z_exner_expl = zero_field(mesh, CellDim, KDim)

    (z_rho_expl_ref, z_exner_expl_ref) = mo_solve_nonhydro_stencil_48_numpy(
        dtime,
        np.asarray(rho_nnow),
        np.asarray(inv_ddqz_z_full),
        np.asarray(z_flxdiv_mass),
        np.asarray(z_contr_w_fl_l),
        np.asarray(exner_pr),
        np.asarray(z_beta),
        np.asarray(z_flxdiv_theta),
        np.asarray(theta_v_ic),
        np.asarray(ddt_exner_phy),
    )

    mo_solve_nonhydro_stencil_48(
        dtime,
        rho_nnow,
        inv_ddqz_z_full,
        z_flxdiv_mass,
        z_contr_w_fl_l,
        exner_pr,
        z_beta,
        z_flxdiv_theta,
        theta_v_ic,
        ddt_exner_phy,
        z_rho_expl,
        z_exner_expl,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(
        np.asarray(z_rho_expl_ref)[:, :-1], np.asarray(z_rho_expl)[:, :-1]
    )
    assert np.allclose(
        np.asarray(z_exner_expl_ref)[:, :-1], np.asarray(z_exner_expl)[:, :-1]
    )
