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

from icon4py.advection.face_val_ppm_stencil_05 import face_val_ppm_stencil_05
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def face_val_ppm_stencil_05_numpy(
    p_cc: np.array,
    p_cellhgt_mc_now: np.array,
    z_slope: np.array,
):
    p_cellhgt_mc_now_k_minus_1 = np.roll(p_cellhgt_mc_now, shift=1, axis=1)
    p_cellhgt_mc_now_k_minus_2 = np.roll(p_cellhgt_mc_now, shift=2, axis=1)
    p_cellhgt_mc_now_k_plus_1 = np.roll(p_cellhgt_mc_now, shift=-1, axis=1)

    p_cc_k_minus_1 = np.roll(p_cc, shift=1, axis=1)
    z_slope_k_minus_1 = np.roll(z_slope, shift=1, axis=1)

    zgeo1 = p_cellhgt_mc_now_k_minus_1 / (p_cellhgt_mc_now_k_minus_1 + p_cellhgt_mc_now)
    zgeo2 = 1.0 / (
        p_cellhgt_mc_now_k_minus_2
        + p_cellhgt_mc_now_k_minus_1
        + p_cellhgt_mc_now
        + p_cellhgt_mc_now_k_plus_1
    )
    zgeo3 = (p_cellhgt_mc_now_k_minus_2 + p_cellhgt_mc_now_k_minus_1) / (
        2.0 * p_cellhgt_mc_now_k_minus_1 + p_cellhgt_mc_now
    )
    zgeo4 = (p_cellhgt_mc_now_k_plus_1 + p_cellhgt_mc_now) / (
        2 * p_cellhgt_mc_now + p_cellhgt_mc_now_k_minus_1
    )

    p_face = (
        p_cc_k_minus_1
        + zgeo1 * (p_cc - p_cc_k_minus_1)
        + zgeo2
        * (
            (2 * p_cellhgt_mc_now * zgeo1) * (zgeo3 - zgeo4) * (p_cc - p_cc_k_minus_1)
            - zgeo3 * p_cellhgt_mc_now_k_minus_1 * z_slope
            + zgeo4 * p_cellhgt_mc_now * z_slope_k_minus_1
        )
    )

    return p_face


def test_face_val_ppm_stencil_05():
    mesh = SimpleMesh()
    p_cc = random_field(mesh, CellDim, KDim)
    p_cellhgt_mc_now = random_field(mesh, CellDim, KDim)
    z_slope = random_field(mesh, CellDim, KDim)
    p_face = zero_field(mesh, CellDim, KDim)

    ref = face_val_ppm_stencil_05_numpy(
        np.asarray(p_cc),
        np.asarray(p_cellhgt_mc_now),
        np.asarray(z_slope),
    )

    face_val_ppm_stencil_05(
        p_cc,
        p_cellhgt_mc_now,
        z_slope,
        p_face,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(ref[:, 1:-1], p_face.__array__()[:, 1:-1])
