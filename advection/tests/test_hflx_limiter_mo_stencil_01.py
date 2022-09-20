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

from icon4py.advection.hflx_limiter_mo_stencil_01 import (
    hflx_limiter_mo_stencil_01,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def hflx_limiter_mo_stencil_01_numpy(
    e2c: np.array,
    p_mflx_tracer_h: np.array,
    p_cc: np.array,
    p_mass_flx_e: np.array,
):
    p_cc_e2c = p_cc[e2c]
    z_mflx_low = 0.5 * (
        p_mass_flx_e * (p_cc_e2c[:, 0] + p_cc_e2c[:, 1])
        - abs(p_mass_flx_e) * (-p_cc_e2c[:, 0] + p_cc_e2c[:, 1])
    )

    z_anti = p_mflx_tracer_h - z_mflx_low

    return z_mflx_low, z_anti


def test_hflx_limiter_mo_stencil_01():
    mesh = SimpleMesh()
    p_mflx_tracer_h = random_field(mesh, EdgeDim, KDim)
    p_cc = random_field(mesh, CellDim, KDim)
    p_mass_flx_e = random_field(mesh, EdgeDim, KDim)
    z_mflx_low = random_field(mesh, EdgeDim, KDim)
    z_anti = random_field(mesh, EdgeDim, KDim)

    z_mflx_low_ref, z_anti_ref = hflx_limiter_mo_stencil_01_numpy(
        mesh.e2c,
        np.asarray(p_mflx_tracer_h),
        np.asarray(p_cc),
        np.asarray(p_mass_flx_e),
    )

    hflx_limiter_mo_stencil_01(
        p_mflx_tracer_h,
        p_cc,
        p_mass_flx_e,
        z_mflx_low,
        z_anti,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(z_mflx_low, z_mflx_low_ref)
    assert np.allclose(z_anti, z_anti_ref)
