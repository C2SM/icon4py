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

from icon4py.model.atmosphere.advection.hflx_limiter_mo_stencil_01a import (
    hflx_limiter_mo_stencil_01a,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def hflx_limiter_mo_stencil_01a_numpy(
    e2c: np.array,
    p_mflx_tracer_h: np.ndarray,
    p_mass_flx_e: np.ndarray,
    p_cc: np.ndarray,
):
    p_cc_e2c = p_cc[e2c]

    z_mflx_low = 0.5 * (
        p_mass_flx_e * (p_cc_e2c[:, 0] + p_cc_e2c[:, 1])
        - np.absolute(p_mass_flx_e) * (p_cc_e2c[:, 1] - p_cc_e2c[:, 0])
    )

    z_anti = p_mflx_tracer_h - z_mflx_low

    return (z_mflx_low, z_anti)


def test_hflx_limiter_mo_stencil_01a():
    grid = SimpleGrid()
    p_mflx_tracer_h = random_field(grid, EdgeDim, KDim)
    p_mass_flx_e = random_field(grid, EdgeDim, KDim)
    p_cc = random_field(grid, CellDim, KDim)
    z_mflx_low = zero_field(grid, EdgeDim, KDim)
    z_anti = zero_field(grid, EdgeDim, KDim)

    ref_1, ref_2 = hflx_limiter_mo_stencil_01a_numpy(
        grid.connectivities[E2CDim],
        np.asarray(p_mflx_tracer_h),
        np.asarray(p_mass_flx_e),
        np.asarray(p_cc),
    )

    hflx_limiter_mo_stencil_01a(
        p_mflx_tracer_h,
        p_mass_flx_e,
        p_cc,
        z_mflx_low,
        z_anti,
        offset_provider={
            "E2C": grid.get_offset_provider["E2C"],
        },
    )

    assert np.allclose(z_mflx_low, ref_1)
    assert np.allclose(z_anti, ref_2)
