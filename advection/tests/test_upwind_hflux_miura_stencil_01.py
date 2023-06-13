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

from icon4py.advection.upwind_hflux_miura_stencil_01 import (
    upwind_hflux_miura_stencil_01,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from .test_utils.simple_mesh import SimpleMesh
from .test_utils.helpers import random_field, zero_field


def upwind_hflux_miura_stencil_01_numpy(
    e2c: np.array,
    p_vn: np.array,
    p_cc: np.array,
    distv_bary_1: np.array,
    distv_bary_2: np.array,
    z_grad_1: np.array,
    z_grad_2: np.array,
    p_mass_flx_e: np.array,
) -> np.array:

    p_cc_e2c = p_cc[e2c]
    z_grad_1_e2c = z_grad_1[e2c]
    z_grad_2_e2c = z_grad_2[e2c]

    p_out_e = np.where(
        p_vn > 0.0,
        (
            p_cc_e2c[:, 0]
            + distv_bary_1 * z_grad_1_e2c[:, 0]
            + distv_bary_2 * z_grad_2_e2c[:, 0]
        )
        * p_mass_flx_e,
        (
            p_cc_e2c[:, 1]
            + distv_bary_1 * z_grad_1_e2c[:, 1]
            + distv_bary_2 * z_grad_2_e2c[:, 1]
        )
        * p_mass_flx_e,
    )

    return p_out_e


def test_upwind_hflux_miura_stencil_01():
    mesh = SimpleMesh()

    p_vn = random_field(mesh, EdgeDim, KDim)
    p_cc = random_field(mesh, CellDim, KDim)
    distv_bary_1 = random_field(mesh, EdgeDim, KDim)
    distv_bary_2 = random_field(mesh, EdgeDim, KDim)
    z_grad_1 = random_field(mesh, CellDim, KDim)
    z_grad_2 = random_field(mesh, CellDim, KDim)
    p_mass_flx_e = random_field(mesh, EdgeDim, KDim)
    p_out_e = zero_field(mesh, EdgeDim, KDim)

    ref = upwind_hflux_miura_stencil_01_numpy(
        mesh.e2c,
        np.asarray(p_vn),
        np.asarray(p_cc),
        np.asarray(distv_bary_1),
        np.asarray(distv_bary_2),
        np.asarray(z_grad_1),
        np.asarray(z_grad_2),
        np.asarray(p_mass_flx_e),
    )

    upwind_hflux_miura_stencil_01(
        p_vn,
        p_cc,
        distv_bary_1,
        distv_bary_2,
        z_grad_1,
        z_grad_2,
        p_mass_flx_e,
        p_out_e,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(p_out_e, ref)
