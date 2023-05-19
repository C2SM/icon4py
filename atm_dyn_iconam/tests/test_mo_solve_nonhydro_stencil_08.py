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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_08 import (
    mo_solve_nonhydro_stencil_08,
)
from icon4py.common.dimension import CellDim, KDim


def mo_solve_nonhydro_stencil_08_numpy(
    wgtfac_c: np.array,
    rho: np.array,
    rho_ref_mc: np.array,
    theta_v: np.array,
    theta_ref_mc: np.array,
) -> tuple[np.array, np.array, np.array]:
    rho_offset_1 = np.roll(rho, shift=1, axis=1)
    rho_ic = wgtfac_c * rho + (1.0 - wgtfac_c) * rho_offset_1
    z_rth_pr_1 = rho - rho_ref_mc
    z_rth_pr_2 = theta_v - theta_ref_mc
    return rho_ic, z_rth_pr_1, z_rth_pr_2


def test_mo_solve_nonhydro_stencil_08():
    mesh = SimpleMesh()

    wgtfac_c = random_field(mesh, CellDim, KDim)
    rho = random_field(mesh, CellDim, KDim)
    rho_ref_mc = random_field(mesh, CellDim, KDim)
    theta_v = random_field(mesh, CellDim, KDim)
    theta_ref_mc = random_field(mesh, CellDim, KDim)

    rho_ic = zero_field(mesh, CellDim, KDim)
    z_rth_pr_1 = zero_field(mesh, CellDim, KDim)
    z_rth_pr_2 = zero_field(mesh, CellDim, KDim)

    rho_ic_ref, z_rth_pr_1_ref, z_rth_pr_2_ref = mo_solve_nonhydro_stencil_08_numpy(
        np.asarray(wgtfac_c),
        np.asarray(rho),
        np.asarray(rho_ref_mc),
        np.asarray(theta_v),
        np.asarray(theta_ref_mc),
    )

    mo_solve_nonhydro_stencil_08(
        wgtfac_c,
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        rho_ic,
        z_rth_pr_1,
        z_rth_pr_2,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(rho_ic[:, 1:], rho_ic_ref[:, 1:])
    assert np.allclose(z_rth_pr_1, z_rth_pr_1_ref)
    assert np.allclose(z_rth_pr_2, z_rth_pr_2_ref)
