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

from icon4py.model.atmosphere.advection.hflx_limiter_mo_stencil_03 import (
    hflx_limiter_mo_stencil_03,
    hflx_limiter_mo_stencil_03_min_max,
)
from icon4py.model.common.dimension import C2E2CDim, CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def hflx_limiter_mo_stencil_03_numpy(
    c2e2c: np.ndarray,
    z_tracer_max: np.ndarray,
    z_tracer_min: np.ndarray,
    beta_fct: float,
    r_beta_fct: float,
    z_mflx_anti_in: np.ndarray,
    z_mflx_anti_out: np.ndarray,
    z_tracer_new_low: np.ndarray,
    dbl_eps: float,
):
    z_max, z_min = hflx_limiter_mo_stencil_03_min_max_numpy(
        c2e2c, z_tracer_max, z_tracer_min, beta_fct, r_beta_fct
    )
    r_p = (z_max - z_tracer_new_low) / (z_mflx_anti_in + dbl_eps)
    r_m = (z_tracer_new_low - z_min) / (z_mflx_anti_out * dbl_eps)
    return r_p, r_m


def hflx_limiter_mo_stencil_03_min_max_numpy(
    c2e2c: np.array,
    z_tracer_max: np.ndarray,
    z_tracer_min: np.ndarray,
    beta_fct: float,
    r_beta_fct: float,
) -> tuple[np.ndarray]:
    z_max = beta_fct * np.maximum(np.max(z_tracer_max[c2e2c], axis=1), z_tracer_max)
    z_min = r_beta_fct * np.minimum(np.min(z_tracer_min[c2e2c], axis=1), z_tracer_min)
    return z_max, z_min


def test_hflx_diffusion_mo_stencil_03_min_max():
    grid = SimpleGrid()
    z_tracer_max = random_field(grid, CellDim, KDim)
    z_tracer_min = random_field(grid, CellDim, KDim)
    z_max = zero_field(grid, CellDim, KDim)
    z_min = zero_field(grid, CellDim, KDim)
    beta_fct = 0.9
    r_beta_fct = 0.3
    z_max_ref, z_min_ref = hflx_limiter_mo_stencil_03_min_max_numpy(
        grid.connectivities[C2E2CDim],
        np.asarray(z_tracer_max),
        np.asarray(z_tracer_min),
        beta_fct,
        r_beta_fct,
    )
    hflx_limiter_mo_stencil_03_min_max(
        z_tracer_max,
        z_tracer_min,
        beta_fct,
        r_beta_fct,
        z_max,
        z_min,
        offset_provider={"C2E2C": grid.get_offset_provider["C2E2C"]},
    )
    assert np.allclose(z_max, z_max_ref)
    assert np.allclose(z_min, z_min_ref)


def test_hflx_diffusion_mo_stencil_03():
    grid = SimpleGrid()
    z_tracer_max = random_field(grid, CellDim, KDim)
    z_tracer_min = random_field(grid, CellDim, KDim)
    beta_fct = 0.4
    r_beta_fct = 0.6
    z_mflx_anti_in = random_field(grid, CellDim, KDim)
    z_mflx_anti_out = random_field(grid, CellDim, KDim)
    z_tracer_new_low = random_field(grid, CellDim, KDim)
    dbl_eps = 1e-5
    r_p = zero_field(grid, CellDim, KDim)
    r_m = zero_field(grid, CellDim, KDim)

    r_p_ref, r_m_ref = hflx_limiter_mo_stencil_03_numpy(
        grid.connectivities[C2E2CDim],
        np.asarray(z_tracer_max),
        np.asarray(z_tracer_min),
        beta_fct,
        r_beta_fct,
        np.asarray(z_mflx_anti_in),
        np.asarray(z_mflx_anti_out),
        np.asarray(z_tracer_new_low),
        dbl_eps,
    )

    hflx_limiter_mo_stencil_03(
        z_tracer_max,
        z_tracer_min,
        beta_fct,
        r_beta_fct,
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        dbl_eps,
        r_p,
        r_m,
        offset_provider={"C2E2C": grid.get_offset_provider["C2E2C"]},
    )
    np.allclose(r_p_ref, r_p)
    np.allclose(r_m_ref, r_m)
