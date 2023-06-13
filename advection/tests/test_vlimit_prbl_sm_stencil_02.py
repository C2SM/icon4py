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

from icon4py.advection.v_limit_prbl_sm_stencil_01 import (
    v_limit_prbl_sm_stencil_01,
)
from icon4py.common.dimension import CellDim, KDim
from .test_utils.simple_mesh import SimpleMesh
from .test_utils.helpers import random_field, zero_field


def v_limit_prbl_sm_stencil_01_numpy(
    p_face: np.array,
    p_cc: np.array,
):

    z_delta = p_face[:, :-1] - p_face[:, 1:]
    z_a6i = 6.0 * (p_cc - 0.5 * (p_face[:, :-1] + p_face[:, 1:]))

    q_face_up, q_face_low = np.where(
        np.abs(z_delta) < -1 * z_a6i,
        np.where(
            (p_cc < np.minimum(p_face[:, :-1], p_face[:, 1:])),
            (p_cc, p_cc),
            np.where(
                p_face[:, :-1] > p_face[:, 1:],
                (3.0 * p_cc - 2.0 * p_face[:, 1:], p_face[:, 1:]),
                (p_face[:, :-1], 3.0 * p_cc - 2.0 * p_face[:, :-1]),
            ),
        ),
        (p_face[:, :-1], p_face[:, 1:]),
    )

    return q_face_up, q_face_low


def test_v_limit_prbl_sm_stencil_01():
    mesh = SimpleMesh()
    p_cc = random_field(mesh, CellDim, KDim)
    p_face = random_field(mesh, CellDim, KDim, extend={KDim: 1})
    p_face_up = zero_field(mesh, CellDim, KDim)
    p_face_low = zero_field(mesh, CellDim, KDim)

    p_face_up_ref, p_face_low_ref = v_limit_prbl_sm_stencil_01_numpy(
        np.asarray(p_face),
        np.asarray(p_cc),
    )

    v_limit_prbl_sm_stencil_01(
        p_face,
        p_cc,
        p_face_up,
        p_face_low,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(p_face_up_ref[:, :-1], p_face_up[:, :-1])
    assert np.allclose(p_face_low_ref[:, :-1], p_face_low[:, :-1])
