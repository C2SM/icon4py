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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_49 import (
    mo_solve_nonhydro_stencil_49,
)
from icon4py.common.dimension import CellDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field


def mo_solve_nonhydro_stencil_49_numpy(
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
) -> tuple[np.array]:
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
    return z_rho_expl, z_exner_expl


def test_mo_solve_nonhydro_stencil_49():
    mesh = SimpleMesh()

    dtime = 7.0

    rho_nnow = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    z_flxdiv_mass = random_field(mesh, CellDim, KDim)
    z_contr_w_fl_l = random_field(mesh, CellDim, KDim, extend={KDim: 1})
    exner_pr = random_field(mesh, CellDim, KDim)
    z_beta = random_field(mesh, CellDim, KDim)
    z_flxdiv_theta = random_field(mesh, CellDim, KDim)
    theta_v_ic = random_field(mesh, CellDim, KDim, extend={KDim: 1})
    ddt_exner_phy = random_field(mesh, CellDim, KDim)

    z_rho_expl = zero_field(mesh, CellDim, KDim)
    z_exner_expl = zero_field(mesh, CellDim, KDim)

    z_rho_expl_ref, z_exner_expl_ref = mo_solve_nonhydro_stencil_49_numpy(
        np.asarray(rho_nnow),
        np.asarray(inv_ddqz_z_full),
        np.asarray(z_flxdiv_mass),
        np.asarray(z_contr_w_fl_l),
        np.asarray(exner_pr),
        np.asarray(z_beta),
        np.asarray(z_flxdiv_theta),
        np.asarray(theta_v_ic),
        np.asarray(ddt_exner_phy),
        dtime,
    )

    mo_solve_nonhydro_stencil_49(
        z_rho_expl,
        z_exner_expl,
        rho_nnow,
        inv_ddqz_z_full,
        z_flxdiv_mass,
        z_contr_w_fl_l,
        exner_pr,
        z_beta,
        z_flxdiv_theta,
        theta_v_ic,
        ddt_exner_phy,
        dtime,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_rho_expl[:, :-1], z_rho_expl_ref[:, :-1])
    assert np.allclose(z_exner_expl[:, :-1], z_exner_expl_ref[:, :-1])
