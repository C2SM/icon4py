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
from functional.ffront.fbuiltins import int32

from icon4py.advection.face_val_ppm_stencil_04 import face_val_ppm_stencil_04
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def face_val_ppm_stencil_04_numpy(
    p_cc: np.array,
    p_cellhgt_mc_now: np.array,
    nudging,
    halo,
    vertical_lower,
    vertical_upper,
) -> np.array:
    p_cellhgt_mc_now_minus_1 = np.roll(p_cellhgt_mc_now, shift=1, axis=1)
    p_cc_minus_1 = np.roll(p_cc, shift=1, axis=1)
    p_face_ref = np.zeros_like(p_cc)

    p_face_ref[nudging:halo, 1:2] = p_cc[nudging:halo, 1:2] * (
        1
        - (
            p_cellhgt_mc_now[nudging:halo, 1:2]
            / p_cellhgt_mc_now_minus_1[nudging:halo, 1:2]
        )
    ) + (
        p_cellhgt_mc_now[nudging:halo, 1:2]
        / (
            p_cellhgt_mc_now_minus_1[nudging:halo, 1:2]
            + p_cellhgt_mc_now[nudging:halo, 1:2]
        )
    ) * (
        (
            p_cellhgt_mc_now[nudging:halo, 1:2]
            / p_cellhgt_mc_now_minus_1[nudging:halo, 1:2]
        )
        * p_cc[nudging:halo, 1:2]
        + p_cc_minus_1[nudging:halo, 1:2]
    )

    p_face_ref[nudging:halo, vertical_upper : vertical_upper + 1] = p_cc[
        nudging:halo, vertical_upper : vertical_upper + 1
    ] * (
        1
        - (
            p_cellhgt_mc_now[nudging:halo, vertical_upper : vertical_upper + 1]
            / p_cellhgt_mc_now_minus_1[
                nudging:halo, vertical_upper : vertical_upper + 1
            ]
        )
    ) + (
        p_cellhgt_mc_now[nudging:halo, vertical_upper : vertical_upper + 1]
        / (
            p_cellhgt_mc_now_minus_1[nudging:halo, vertical_upper : vertical_upper + 1]
            + p_cellhgt_mc_now[nudging:halo, vertical_upper : vertical_upper + 1]
        )
    ) * (
        (
            p_cellhgt_mc_now[nudging:halo, vertical_upper : vertical_upper + 1]
            / p_cellhgt_mc_now_minus_1[
                nudging:halo, vertical_upper : vertical_upper + 1
            ]
        )
        * p_cc[nudging:halo, vertical_upper : vertical_upper + 1]
        + p_cc_minus_1[nudging:halo, vertical_upper : vertical_upper + 1]
    )

    p_face_ref[nudging:halo, 0:1] = p_cc[nudging:halo, 0:1]
    return p_face_ref


def test_face_val_ppm_stencil_04():
    mesh = SimpleMesh()

    p_cc = random_field(mesh, CellDim, KDim)
    p_cellhgt_mc_now = random_field(mesh, CellDim, KDim)
    p_face = zero_field(mesh, CellDim, KDim)
    nudging = int32(4)
    halo = int32(16)
    vertical_lower = int32(0)
    vertical_upper = int32(np.asarray(p_face).shape[1])

    p_face_ref = face_val_ppm_stencil_04_numpy(
        np.asarray(p_cc),
        np.asarray(p_cellhgt_mc_now),
        nudging,
        halo,
        vertical_lower,
        vertical_upper,
    )
    face_val_ppm_stencil_04(
        p_cc,
        p_cellhgt_mc_now,
        p_face,
        nudging,
        halo,
        vertical_lower,
        vertical_upper,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(p_face_ref, p_face)
