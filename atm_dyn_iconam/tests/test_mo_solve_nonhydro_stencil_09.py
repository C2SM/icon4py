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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_09 import (
    mo_solve_nonhydro_stencil_09,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_09_numpy(
    wgtfac_c: np.array,
    z_rth_pr_2: np.array,
    theta_v: np.array,
    vwind_expl_wgt: np.array,
    exner_pr: np.array,
    d_exner_dz_ref_ic: np.array,
    ddqz_z_half: np.array,
) -> tuple[np.array, np.array, np.array]:
    z_rth_pr_2_offset = np.roll(z_rth_pr_2, axis=1, shift=1)
    theta_v_offset = np.roll(theta_v, axis=1, shift=1)
    exner_pr_offset = np.roll(exner_pr, axis=1, shift=1)
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)

    z_theta_v_pr_ic = wgtfac_c * z_rth_pr_2 + (1.0 - wgtfac_c) * z_rth_pr_2_offset
    theta_v_ic = wgtfac_c * theta_v + (1 - wgtfac_c) * theta_v_offset
    z_th_ddz_exner_c = (
        vwind_expl_wgt * theta_v_ic * (exner_pr_offset - exner_pr) / ddqz_z_half
        + z_theta_v_pr_ic * d_exner_dz_ref_ic
    )
    return z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c


def test_mo_solve_nonhydro_stencil_09():
    mesh = SimpleMesh()

    wgtfac_c = random_field(mesh, CellDim, KDim)
    z_rth_pr_2 = random_field(mesh, CellDim, KDim)
    theta_v = random_field(mesh, CellDim, KDim)
    vwind_expl_wgt = random_field(mesh, CellDim)
    exner_pr = random_field(mesh, CellDim, KDim)
    d_exner_dz_ref_ic = random_field(mesh, CellDim, KDim)
    ddqz_z_half = random_field(mesh, CellDim, KDim)

    z_theta_v_pr_ic = zero_field(mesh, CellDim, KDim)
    theta_v_ic = zero_field(mesh, CellDim, KDim)
    z_th_ddz_exner_c = zero_field(mesh, CellDim, KDim)

    (
        z_theta_v_pr_ic_ref,
        theta_v_ic_ref,
        z_th_ddz_exner_c_ref,
    ) = mo_solve_nonhydro_stencil_09_numpy(
        np.asarray(wgtfac_c),
        np.asarray(z_rth_pr_2),
        np.asarray(theta_v),
        np.asarray(vwind_expl_wgt),
        np.asarray(exner_pr),
        np.asarray(d_exner_dz_ref_ic),
        np.asarray(ddqz_z_half),
    )

    mo_solve_nonhydro_stencil_09(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_theta_v_pr_ic[:, 1:], z_theta_v_pr_ic_ref[:, 1:])
    assert np.allclose(theta_v_ic[:, 1:], theta_v_ic_ref[:, 1:])
    assert np.allclose(z_th_ddz_exner_c[:, 1:], z_th_ddz_exner_c_ref[:, 1:])
