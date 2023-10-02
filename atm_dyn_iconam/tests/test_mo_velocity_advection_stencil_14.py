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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_14 import (
    mo_velocity_advection_stencil_14,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, random_mask, zero_field


def mo_velocity_advection_stencil_14_numpy(
    ddqz_z_half: np.array,
    z_w_con_c: np.array,
    cfl_w_limit,
    dtime,
) -> tuple[np.array]:
    num_rows, num_cols = z_w_con_c.shape
    cfl_clipping = np.where(
        np.abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
        np.ones([num_rows, num_cols]),
        np.zeros_like(z_w_con_c),
    )

    vcfl = np.where(cfl_clipping == 1.0, z_w_con_c * dtime / ddqz_z_half, 0.0)
    z_w_con_c = np.where(
        (cfl_clipping == 1.0) & (vcfl < -0.85), -0.85 * ddqz_z_half / dtime, z_w_con_c
    )
    z_w_con_c = np.where(
        (cfl_clipping == 1.0) & (vcfl > 0.85), 0.85 * ddqz_z_half / dtime, z_w_con_c
    )

    return cfl_clipping, vcfl, z_w_con_c


def test_mo_velocity_advection_stencil_14():
    mesh = SimpleMesh()

    ddqz_z_half = random_field(mesh, CellDim, KDim)
    z_w_con_c = random_field(mesh, CellDim, KDim)
    cfl_clipping = random_mask(mesh, CellDim, KDim, dtype=bool)
    vcfl = zero_field(mesh, CellDim, KDim)
    cfl_w_limit = 5.0
    dtime = 9.0

    (
        cfl_clipping_ref,
        vcfl_ref,
        z_w_con_c_ref,
    ) = mo_velocity_advection_stencil_14_numpy(
        np.asarray(ddqz_z_half),
        np.asarray(z_w_con_c),
        cfl_w_limit,
        dtime,
    )

    mo_velocity_advection_stencil_14(
        ddqz_z_half,
        z_w_con_c,
        cfl_clipping,
        vcfl,
        cfl_w_limit,
        dtime,
        offset_provider={},
    )
    assert np.allclose(cfl_clipping, cfl_clipping_ref)
    assert np.allclose(vcfl, vcfl_ref)
    assert np.allclose(z_w_con_c, z_w_con_c_ref)
