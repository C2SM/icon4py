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

from icon4py.model.atmosphere.advection.btraj_dreg_stencil_01 import btraj_dreg_stencil_01
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def btraj_dreg_stencil_01_numpy(
    lcounterclock: bool,
    p_vn: np.array,
    tangent_orientation: np.array,
):
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

    tangent_orientation = np.broadcast_to(tangent_orientation, p_vn.shape)

    lvn_sys_pos_true = np.where(tangent_orientation * p_vn >= 0.0, True, False)

    mask_lcounterclock = np.broadcast_to(lcounterclock, p_vn.shape)

    lvn_sys_pos = np.where(mask_lcounterclock, lvn_sys_pos_true, False)

    return lvn_sys_pos


def test_btraj_dreg_stencil_01():
    grid = SimpleGrid()
    lcounterclock = True
    p_vn = random_field(grid, EdgeDim, KDim)

    tangent_orientation = random_field(grid, EdgeDim)

    lvn_sys_pos = zero_field(grid, EdgeDim, KDim, dtype=bool)

    ref = btraj_dreg_stencil_01_numpy(
        lcounterclock,
        np.asarray(p_vn),
        np.asarray(tangent_orientation),
    )

    btraj_dreg_stencil_01.with_backend(roundtrip.backend)(
        lcounterclock,
        p_vn,
        tangent_orientation,
        lvn_sys_pos,
        offset_provider={},
    )

    assert np.allclose(ref, lvn_sys_pos)
