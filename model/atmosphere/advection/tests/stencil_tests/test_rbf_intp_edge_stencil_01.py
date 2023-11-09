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

from icon4py.model.atmosphere.advection.rbf_intp_edge_stencil_01 import rbf_intp_edge_stencil_01
from icon4py.model.common.dimension import E2C2EDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def rbf_intp_edge_stencil_01_numpy(
    e2c2e: np.array,
    p_vn_in: np.array,
    ptr_coeff: np.array,
) -> np.array:
    ptr_coeff = np.expand_dims(ptr_coeff, axis=-1)
    p_vt_out = np.sum(p_vn_in[e2c2e] * ptr_coeff, axis=1)
    return p_vt_out


def test_rbf_intp_edge_stencil_01():
    grid = SimpleGrid()

    p_vn_in = random_field(grid, EdgeDim, KDim)
    ptr_coeff = random_field(grid, EdgeDim, E2C2EDim)
    p_vt_out = zero_field(grid, EdgeDim, KDim)

    ref = rbf_intp_edge_stencil_01_numpy(
        grid.connectivities[E2C2EDim],
        np.asarray(p_vn_in),
        np.asarray(ptr_coeff),
    )

    rbf_intp_edge_stencil_01(
        p_vn_in,
        ptr_coeff,
        p_vt_out,
        offset_provider={
            "E2C2E": grid.get_offset_provider("E2C2E"),
        },
    )
    assert np.allclose(p_vt_out, ref)
