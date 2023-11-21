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

from icon4py.model.atmosphere.advection.set_zero_c import set_zero_c
from icon4py.model.common.dimension import CellDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def test_set_zero_cell_k():
    grid = SimpleGrid()
    field = random_field(grid, CellDim)

    set_zero_c.with_backend(roundtrip.backend)(field, offset_provider={})
    assert np.allclose(field, zero_field(grid, CellDim))
