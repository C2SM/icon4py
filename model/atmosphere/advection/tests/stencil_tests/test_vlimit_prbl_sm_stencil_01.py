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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.v_limit_prbl_sm_stencil_01 import v_limit_prbl_sm_stencil_01
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def v_limit_prbl_sm_stencil_01_numpy(
    p_face: np.array,
    p_cc: np.array,
):

    z_delta = p_face[:, :-1] - p_face[:, 1:]
    z_a6i = 6.0 * (p_cc - 0.5 * (p_face[:, :-1] + p_face[:, 1:]))

    l_limit = np.where(np.abs(z_delta) < -1 * z_a6i, int32(1), int32(0))

    return l_limit


def test_v_limit_prbl_sm_stencil_01(backend):
    grid = SimpleGrid()
    p_cc = random_field(grid, CellDim, KDim)
    p_face = random_field(grid, CellDim, KDim, extend={KDim: 1})
    l_limit = zero_field(grid, CellDim, KDim, dtype=int32)

    l_limit_ref = v_limit_prbl_sm_stencil_01_numpy(
        np.asarray(p_face),
        np.asarray(p_cc),
    )

    v_limit_prbl_sm_stencil_01.with_backend(backend)(
        p_face,
        p_cc,
        l_limit,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(l_limit_ref, l_limit)
