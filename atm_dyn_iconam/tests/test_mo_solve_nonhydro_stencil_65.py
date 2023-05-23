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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_65 import (
    mo_solve_nonhydro_stencil_65,
)
from icon4py.common.dimension import CellDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field


def mo_solve_nonhydro_stencil_65_numpy(
    rho_ic: np.array,
    vwind_expl_wgt: np.array,
    vwind_impl_wgt: np.array,
    w_now: np.array,
    w_new: np.array,
    w_concorr_c: np.array,
    mass_flx_ic: np.array,
    r_nsubsteps: float,
) -> np.array:
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
    mass_flx_ic = mass_flx_ic + (
        r_nsubsteps
        * rho_ic
        * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c)
    )
    return mass_flx_ic


def test_mo_solve_nonhydro_stencil_65():
    mesh = SimpleMesh()

    r_nsubsteps = 10.0
    rho_ic = random_field(mesh, CellDim, KDim)
    vwind_expl_wgt = random_field(mesh, CellDim)
    vwind_impl_wgt = random_field(mesh, CellDim)
    w_now = random_field(mesh, CellDim, KDim)
    w_new = random_field(mesh, CellDim, KDim)
    w_concorr_c = random_field(mesh, CellDim, KDim)
    mass_flx_ic = random_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_65_numpy(
        np.asarray(rho_ic),
        np.asarray(vwind_expl_wgt),
        np.asarray(vwind_impl_wgt),
        np.asarray(w_now),
        np.asarray(w_new),
        np.asarray(w_concorr_c),
        np.asarray(mass_flx_ic),
        r_nsubsteps,
    )

    mo_solve_nonhydro_stencil_65(
        rho_ic,
        vwind_expl_wgt,
        vwind_impl_wgt,
        w_now,
        w_new,
        w_concorr_c,
        mass_flx_ic,
        r_nsubsteps,
        offset_provider={},
    )
    assert np.allclose(mass_flx_ic, ref)
