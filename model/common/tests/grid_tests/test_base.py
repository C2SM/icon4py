# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, gridfile, simple
from icon4py.model.common.utils import data_allocation as data_alloc


def test_replace_skip_values(grid_file, caplog, backend):
    grid = simple.SimpleGrid()
    domain = (dims.CellDim, dims.C2E2CDim)
    xp = data_alloc.array_ns(backend)
    neighbor_table = data_alloc.random_field(
        grid, *domain, low=0, high=grid.num_cells, dtype=gtx.int32, backend=backend
    ).ndarray
    neighbor_table[0, 1:] = gridfile.GridFile.INVALID_INDEX

    assert xp.any(neighbor_table == gridfile.GridFile.INVALID_INDEX)
    neighbor_table = base.replace_skip_values(domain, neighbor_table, array_ns=xp)
    assert not xp.any(neighbor_table == gridfile.GridFile.INVALID_INDEX)
