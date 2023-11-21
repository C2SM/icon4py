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
from gt4py.next.program_processors.runners import roundtrip

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02a import face_val_ppm_stencil_02a
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field


def face_val_ppm_stencil_02a_numpy(
    p_cc: np.array,
    p_cellhgt_mc_now: np.array,
):

    p_face = p_cc.copy()

    p_face[:, 1:] = p_cc[:, 1:] * (1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])) + (
        p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])
    ) * ((p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1])

    return p_face


def test_face_val_ppm_stencil_02a():
    grid = SimpleGrid()
    p_cc = random_field(grid, CellDim, KDim)
    p_cellhgt_mc_now = random_field(grid, CellDim, KDim)
    p_face = random_field(grid, CellDim, KDim)

    ref = face_val_ppm_stencil_02a_numpy(
        np.asarray(p_cc),
        np.asarray(p_cellhgt_mc_now),
    )

    face_val_ppm_stencil_02a.with_backend(roundtrip.backend)(
        p_cc,
        p_cellhgt_mc_now,
        p_face,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(ref[:, 1:], p_face[:, 1:])
