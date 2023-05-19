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
from simple_mesh import SimpleMesh
from utils import random_field, random_mask

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_68 import (
    mo_solve_nonhydro_stencil_68,
)
from icon4py.common.dimension import CellDim, KDim


def mo_solve_nonhydro_stencil_68_numpy(
    mask_prog_halo_c: np.array,
    rho_now: np.array,
    theta_v_now: np.array,
    exner_new: np.array,
    exner_now: np.array,
    rho_new: np.array,
    theta_v_new: np.array,
    cvd_o_rd: float,
) -> np.array:

    mask_prog_halo_c = np.expand_dims(mask_prog_halo_c, axis=-1)

    theta_v_new = np.where(
        mask_prog_halo_c,
        rho_now
        * theta_v_now
        * ((exner_new / exner_now - 1) * cvd_o_rd + 1.0)
        / rho_new,
        theta_v_new,
    )
    return theta_v_new


def test_mo_solve_nonhydro_stencil_68():
    mesh = SimpleMesh()

    mask_prog_halo_c = random_mask(mesh, CellDim)
    rho_now = random_field(mesh, CellDim, KDim)
    theta_v_now = random_field(mesh, CellDim, KDim)
    exner_new = random_field(mesh, CellDim, KDim)
    exner_now = random_field(mesh, CellDim, KDim)
    rho_new = random_field(mesh, CellDim, KDim)
    theta_v_new = random_field(mesh, CellDim, KDim)
    cvd_o_rd = 10.0

    ref = mo_solve_nonhydro_stencil_68_numpy(
        np.asarray(mask_prog_halo_c),
        np.asarray(rho_now),
        np.asarray(theta_v_now),
        np.asarray(exner_new),
        np.asarray(exner_now),
        np.asarray(rho_new),
        np.asarray(theta_v_new),
        np.asarray(cvd_o_rd),
    )
    mo_solve_nonhydro_stencil_68(
        mask_prog_halo_c,
        rho_now,
        theta_v_now,
        exner_new,
        exner_now,
        rho_new,
        theta_v_new,
        cvd_o_rd,
        offset_provider={},
    )
    assert np.allclose(theta_v_new, ref)
