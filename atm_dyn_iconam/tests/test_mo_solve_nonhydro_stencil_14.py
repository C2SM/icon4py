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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_14 import (
    mo_solve_nonhydro_stencil_14,
)
from icon4py.common.dimension import EdgeDim, KDim

from .test_utils.helpers import zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_14_numpy(
    z_rho_e: np.array, z_theta_v_e: np.array
) -> tuple[np.array]:
    z_rho_e = np.zeros_like(z_rho_e)
    z_theta_v_e = np.zeros_like(z_theta_v_e)
    return z_rho_e, z_theta_v_e


def test_mo_solve_nonhydro_stencil_14_z_theta_v_e():
    mesh = SimpleMesh()

    z_rho_e = zero_field(mesh, EdgeDim, KDim)
    z_theta_v_e = zero_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_14_numpy(
        np.asarray(z_rho_e), np.asarray(z_theta_v_e)
    )
    mo_solve_nonhydro_stencil_14(
        z_rho_e,
        z_theta_v_e,
        offset_provider={},
    )
    assert np.allclose(z_theta_v_e, ref)
