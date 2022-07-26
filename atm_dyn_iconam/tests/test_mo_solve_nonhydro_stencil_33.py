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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_33 import (
    mo_solve_nonhydro_stencil_33,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import zero_field


def mo_solve_nonhydro_stencil_33_vn_traj_numpy(vn_traj: np.array) -> np.array:
    vn_traj = np.zeros_like(vn_traj)
    return vn_traj


def mo_solve_nonhydro_stencil_33_mass_flx_me_numpy(mass_flx_me: np.array) -> np.array:
    mass_flx_me = np.zeros_like(mass_flx_me)
    return mass_flx_me


def mo_solve_nonhydro_stencil_33_numpy(
    vn_traj: np.array, mass_flx_me: np.array
) -> tuple[np.array]:
    vn_traj = mo_solve_nonhydro_stencil_33_vn_traj_numpy(vn_traj)
    mass_flx_me = mo_solve_nonhydro_stencil_33_mass_flx_me_numpy(mass_flx_me)
    return vn_traj, mass_flx_me


def test_mo_solve_nonhydro_stencil_33():
    mesh = SimpleMesh()

    vn_traj = zero_field(mesh, EdgeDim, KDim)
    mass_flx_me = zero_field(mesh, EdgeDim, KDim)

    vn_traj_ref, mass_flx_me_ref = mo_solve_nonhydro_stencil_33_numpy(
        np.asarray(vn_traj), np.asarray(mass_flx_me)
    )
    mo_solve_nonhydro_stencil_33(
        vn_traj,
        mass_flx_me,
        offset_provider={},
    )

    assert np.allclose(vn_traj, vn_traj_ref)
    assert np.allclose(mass_flx_me, mass_flx_me_ref)
