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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_66 import (
    mo_solve_nonhydro_stencil_66,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, random_mask


def mo_solve_nonhydro_stencil_66_numpy(
    bdy_halo_c: np.array,
    rho: np.array,
    theta_v: np.array,
    exner: np.array,
    rd_o_cvd: float,
    rd_o_p0ref: float,
) -> tuple[np.array]:
    bdy_halo_c = np.expand_dims(bdy_halo_c, axis=-1)

    theta_v = np.where(bdy_halo_c == 1, exner, theta_v)
    exner = np.where(
        bdy_halo_c == 1, np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * exner)), exner
    )

    return theta_v, exner


def test_mo_solve_nonhydro_stencil_66():
    mesh = SimpleMesh()

    rd_o_cvd = 10.0
    rd_o_p0ref = 20.0
    bdy_halo_c = random_mask(mesh, CellDim)
    exner = random_field(mesh, CellDim, KDim, low=1, high=2)
    rho = random_field(mesh, CellDim, KDim, low=1, high=2)
    theta_v = random_field(mesh, CellDim, KDim, low=1, high=2)

    theta_v_ref, exner_ref = mo_solve_nonhydro_stencil_66_numpy(
        np.asarray(bdy_halo_c),
        np.asarray(rho),
        np.asarray(theta_v),
        np.asarray(exner),
        rd_o_cvd,
        rd_o_p0ref,
    )

    mo_solve_nonhydro_stencil_66(
        bdy_halo_c,
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        offset_provider={},
    )

    assert np.allclose(theta_v, theta_v_ref)
    assert np.allclose(exner, exner_ref)
