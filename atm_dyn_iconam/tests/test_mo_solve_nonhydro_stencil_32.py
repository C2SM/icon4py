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
from utils.helpers import random_field, zero_field
from utils.simple_mesh import SimpleMesh

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_32 import (
    mo_solve_nonhydro_stencil_32,
)
from icon4py.common.dimension import EdgeDim, KDim


def mo_solve_nonhydro_stencil_32_numpy(
    z_rho_e: np.array,
    z_vn_avg: np.array,
    ddqz_z_full_e: np.array,
    z_theta_v_e: np.array,
) -> tuple[np.array]:
    mass_fl_e = z_rho_e * z_vn_avg * ddqz_z_full_e
    z_theta_v_fl_e = mass_fl_e * z_theta_v_e
    return mass_fl_e, z_theta_v_fl_e


def test_mo_solve_nonhydro_stencil_32():
    mesh = SimpleMesh()

    z_rho_e = random_field(mesh, EdgeDim, KDim)
    z_vn_avg = random_field(mesh, EdgeDim, KDim)
    ddqz_z_full_e = random_field(mesh, EdgeDim, KDim)
    mass_fl_e = random_field(mesh, EdgeDim, KDim)
    z_theta_v_e = random_field(mesh, EdgeDim, KDim)
    z_theta_v_fl_e = zero_field(mesh, EdgeDim, KDim)

    mass_fl_e_ref, z_theta_v_fl_e_ref = mo_solve_nonhydro_stencil_32_numpy(
        np.asarray(z_rho_e),
        np.asarray(z_vn_avg),
        np.asarray(ddqz_z_full_e),
        np.asarray(z_theta_v_e),
    )

    mo_solve_nonhydro_stencil_32(
        z_rho_e,
        z_vn_avg,
        ddqz_z_full_e,
        z_theta_v_e,
        mass_fl_e,
        z_theta_v_fl_e,
        offset_provider={},
    )

    assert np.allclose(mass_fl_e, mass_fl_e_ref)
    assert np.allclose(z_theta_v_fl_e, z_theta_v_fl_e_ref)
