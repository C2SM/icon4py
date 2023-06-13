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

from icon4py.advection.hflx_limiter_mo_stencil_04 import (
    hflx_limiter_mo_stencil_04,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def hflx_limiter_mo_stencil_04_numpy(
    e2c: np.ndarray,
    z_anti: np.ndarray,
    r_m: np.ndarray,
    r_p: np.ndarray,
    z_mflx_low: np.ndarray,
):
    r_frac = np.where(
        z_anti >= 0,
        np.minimum(r_m[e2c[:, 0]], r_p[e2c[:, 1]]),
        np.minimum(r_m[e2c[:, 1]], r_p[e2c[:, 0]]),
    )
    return z_mflx_low + np.minimum(1.0, r_frac) * z_anti


def test_hflx_limiter_mo_stencil_04():
    mesh = SimpleMesh()
    z_anti = random_field(mesh, EdgeDim, KDim, low=-2.0, high=2.0)
    r_m = random_field(mesh, CellDim, KDim)
    r_p = random_field(mesh, CellDim, KDim)
    z_mflx_low = random_field(mesh, EdgeDim, KDim)
    p_mflx_tracer_h = zero_field(mesh, EdgeDim, KDim)
    ref = hflx_limiter_mo_stencil_04_numpy(
        mesh.e2c,
        np.asarray(z_anti),
        np.asarray(r_m),
        np.asarray(r_p),
        np.asarray(z_mflx_low),
    )
    hflx_limiter_mo_stencil_04(
        z_anti,
        r_m,
        r_p,
        z_mflx_low,
        p_mflx_tracer_h,
        offset_provider={"E2C": mesh.get_e2c_offset_provider()},
    )
    assert np.allclose(p_mflx_tracer_h, ref)
