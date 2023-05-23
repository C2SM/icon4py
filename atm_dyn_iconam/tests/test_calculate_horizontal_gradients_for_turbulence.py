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

from icon4py.atm_dyn_iconam.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
)
from icon4py.common.dimension import C2E2CODim, CellDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field


def calculate_horizontal_gradients_for_turbulence_numpy(
    c2e2c0: np.array,
    w: np.array,
    geofac_grg_x: np.array,
    geofac_grg_y: np.array,
) -> tuple[np.array]:
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    dwdx = np.sum(geofac_grg_x * w[c2e2c0], axis=1)

    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    dwdy = np.sum(geofac_grg_y * w[c2e2c0], axis=1)
    return dwdx, dwdy


def test_calculate_horizontal_gradients_for_turbulence():
    mesh = SimpleMesh()

    w = random_field(mesh, CellDim, KDim)
    geofac_grg_x = random_field(mesh, CellDim, C2E2CODim)
    geofac_grg_y = random_field(mesh, CellDim, C2E2CODim)
    dwdx = zero_field(mesh, CellDim, KDim)
    dwdy = zero_field(mesh, CellDim, KDim)

    dwdx_ref, dwdy_ref = calculate_horizontal_gradients_for_turbulence_numpy(
        mesh.c2e2cO, np.asarray(w), np.asarray(geofac_grg_x), np.asarray(geofac_grg_y)
    )
    calculate_horizontal_gradients_for_turbulence(
        w,
        geofac_grg_x,
        geofac_grg_y,
        dwdx,
        dwdy,
        offset_provider={"C2E2CO": mesh.get_c2e2cO_offset_provider()},
    )

    assert np.allclose(dwdx, dwdx_ref)
    assert np.allclose(dwdy, dwdy_ref)
