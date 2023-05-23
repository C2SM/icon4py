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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_34 import (
    mo_solve_nonhydro_stencil_34,
)
from icon4py.common.dimension import EdgeDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field


def mo_solve_nonhydro_stencil_34_numpy(
    z_vn_avg: np.array,
    mass_fl_e: np.array,
    vn_traj: np.array,
    mass_flx_me: np.array,
    r_nsubsteps,
) -> tuple[np.array]:
    vn_traj = vn_traj + r_nsubsteps * z_vn_avg
    mass_flx_me = mass_flx_me + r_nsubsteps * mass_fl_e
    return vn_traj, mass_flx_me


def test_mo_solve_nonhydro_stencil_34():
    mesh = SimpleMesh()

    mass_fl_e = random_field(mesh, EdgeDim, KDim)
    mass_flx_me = random_field(mesh, EdgeDim, KDim)
    z_vn_avg = random_field(mesh, EdgeDim, KDim)
    vn_traj = random_field(mesh, EdgeDim, KDim)
    r_nsubsteps = 9.0

    vn_traj_ref, mass_flx_me_ref = mo_solve_nonhydro_stencil_34_numpy(
        np.asarray(z_vn_avg),
        np.asarray(mass_fl_e),
        np.asarray(vn_traj),
        np.asarray(mass_flx_me),
        r_nsubsteps,
    )

    mo_solve_nonhydro_stencil_34(
        z_vn_avg,
        mass_fl_e,
        vn_traj,
        mass_flx_me,
        r_nsubsteps,
        offset_provider={},
    )
    assert np.allclose(vn_traj_ref, vn_traj)
    assert np.allclose(mass_flx_me_ref, mass_flx_me)
