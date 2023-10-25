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

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02b import face_val_ppm_stencil_02b
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.common.grid.simple import SimpleGrid


def face_val_ppm_stencil_02b_numpy(
    p_cc: np.array,
):

    p_face = p_cc.copy()

    return p_face


def test_face_val_ppm_stencil_02b():
    grid = SimpleGrid()
    p_cc = random_field(grid, CellDim, KDim)
    p_face = random_field(grid, CellDim, KDim)

    ref = face_val_ppm_stencil_02b_numpy(
        np.asarray(p_cc),
    )

    face_val_ppm_stencil_02b(
        p_cc,
        p_face,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(ref, p_face)
