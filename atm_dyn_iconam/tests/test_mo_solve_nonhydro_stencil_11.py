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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_11 import (
    mo_solve_nonhydro_stencil_11,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_11_numpy(
    wgtfacq_c: np.array,
    z_rth_pr: np.array,
    theta_ref_ic: np.array,
    z_theta_v_pr_ic: np.array,
) -> tuple[np.array, np.array]:
    z_theta_v_pr_ic = (
        np.roll(wgtfacq_c, shift=1, axis=1) * np.roll(z_rth_pr, shift=1, axis=1)
        + np.roll(wgtfacq_c, shift=2, axis=1) * np.roll(z_rth_pr, shift=2, axis=1)
        + np.roll(wgtfacq_c, shift=3, axis=1) * np.roll(z_rth_pr, shift=3, axis=1)
    )
    theta_v_ic = theta_ref_ic + z_theta_v_pr_ic
    return z_theta_v_pr_ic, theta_v_ic


def test_mo_solve_nonhydro_stencil_11():
    mesh = SimpleMesh()

    wgtfacq_c = random_field(mesh, CellDim, KDim)
    z_rth_pr = random_field(mesh, CellDim, KDim)
    theta_ref_ic = random_field(mesh, CellDim, KDim)
    z_theta_v_pr_ic = random_field(mesh, CellDim, KDim)

    theta_v_ic = zero_field(mesh, CellDim, KDim)

    z_theta_v_pr_ic_ref, theta_v_ic_ref = mo_solve_nonhydro_stencil_11_numpy(
        np.asarray(wgtfacq_c),
        np.asarray(z_rth_pr),
        np.asarray(theta_ref_ic),
        np.asarray(z_theta_v_pr_ic),
    )

    mo_solve_nonhydro_stencil_11(
        wgtfacq_c,
        z_rth_pr,
        theta_ref_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_theta_v_pr_ic[:, 3:], z_theta_v_pr_ic_ref[:, 3:])
    assert np.allclose(theta_v_ic, theta_v_ic_ref)
