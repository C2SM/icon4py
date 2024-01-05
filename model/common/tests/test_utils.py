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

from icon4py.model.common.dimension import KDim, VertexDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.utils import scale_k, set_zero_v_k


def test_scale_k():
    grid = SimpleGrid()
    field = random_field(grid, KDim)
    scaled_field = zero_field(grid, KDim)
    factor = 2.0
    scale_k(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * field.asnumpy(), scaled_field.asnumpy())


def test_set_zero_vertex_k(backend):
    grid = SimpleGrid()
    f = random_field(grid, VertexDim, KDim)
    set_zero_v_k(f, offset_provider={})
    assert np.allclose(0.0, f.asnumpy())
