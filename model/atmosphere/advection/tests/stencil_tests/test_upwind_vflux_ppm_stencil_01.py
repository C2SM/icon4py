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

from icon4py.model.atmosphere.advection.upwind_vflux_ppm_stencil_01 import (
    upwind_vflux_ppm_stencil_01,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def upwind_vflux_ppm_stencil_01_numpy(
    z_face_up: np.ndarray, z_face_low: np.ndarray, p_cc: np.ndarray
) -> tuple[np.ndarray]:
    z_delta_q = 0.5 * (z_face_up - z_face_low)
    z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)
    return z_delta_q, z_a1


def test_upwind_vflux_ppm_stencil_01(backend):
    grid = SimpleGrid()
    z_face_up = random_field(grid, CellDim, KDim)
    z_face_down = random_field(grid, CellDim, KDim)
    p_cc = random_field(grid, CellDim, KDim)
    z_delta_q = zero_field(grid, CellDim, KDim)
    z_a1 = zero_field(grid, CellDim, KDim)

    ref_z_delta_q, ref_z_a1 = upwind_vflux_ppm_stencil_01_numpy(
        np.asarray(z_face_up), np.asarray(z_face_down), np.asarray(p_cc)
    )

    upwind_vflux_ppm_stencil_01.with_backend(backend)(
        z_face_up, z_face_down, p_cc, z_delta_q, z_a1, offset_provider={}
    )

    assert np.allclose(ref_z_delta_q, z_delta_q)
    assert np.allclose(ref_z_a1, z_a1)
