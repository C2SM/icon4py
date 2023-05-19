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
from utils.helpers import random_field, random_mask
from utils.simple_mesh import SimpleMesh

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_18 import (
    mo_velocity_advection_stencil_18,
)
from icon4py.common.dimension import C2E2CODim, CellDim, KDim


def mo_velocity_advection_stencil_18_numpy(
    c2e2c0: np.array,
    levelmask: np.array,
    cfl_clipping: np.array,
    owner_mask: np.array,
    z_w_con_c: np.array,
    ddqz_z_half: np.array,
    area: np.array,
    geofac_n2s: np.array,
    w: np.array,
    ddt_w_adv: np.array,
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
):
    levelmask = np.expand_dims(levelmask, axis=0)
    owner_mask = np.expand_dims(owner_mask, axis=-1)
    area = np.expand_dims(area, axis=-1)
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

    difcoef = np.where(
        (levelmask == 1) & (cfl_clipping == 1) & (owner_mask == 1),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(z_w_con_c) * dtime / ddqz_z_half - cfl_w_limit * dtime,
        ),
        0,
    )

    ddt_w_adv = np.where(
        (levelmask == 1) & (cfl_clipping == 1) & (owner_mask == 1),
        ddt_w_adv + difcoef * area * np.sum(w[c2e2c0] * geofac_n2s, axis=1),
        ddt_w_adv,
    )

    return ddt_w_adv


def test_mo_velocity_advection_stencil_18():
    mesh = SimpleMesh()

    levelmask = random_mask(mesh, KDim)
    cfl_clipping = random_mask(mesh, CellDim, KDim)
    owner_mask = random_mask(mesh, CellDim)
    z_w_con_c = random_field(mesh, CellDim, KDim)
    ddqz_z_half = random_field(mesh, CellDim, KDim)
    area = random_field(mesh, CellDim)
    geofac_n2s = random_field(mesh, CellDim, C2E2CODim)
    w = random_field(mesh, CellDim, KDim)
    ddt_w_adv = random_field(mesh, CellDim, KDim)
    scalfac_exdiff = 10.0
    cfl_w_limit = 3.0
    dtime = 2.0

    ref = mo_velocity_advection_stencil_18_numpy(
        mesh.c2e2cO,
        np.asarray(levelmask),
        np.asarray(cfl_clipping),
        np.asarray(owner_mask),
        np.asarray(z_w_con_c),
        np.asarray(ddqz_z_half),
        np.asarray(area),
        np.asarray(geofac_n2s),
        np.asarray(w),
        np.asarray(ddt_w_adv),
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
    )

    mo_velocity_advection_stencil_18(
        levelmask,
        cfl_clipping,
        owner_mask,
        z_w_con_c,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        ddt_w_adv,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        offset_provider={"C2E2CO": mesh.get_c2e2cO_offset_provider()},
    )

    assert np.allclose(ddt_w_adv, ref)
