# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid


def test_domain():
    domain = h_grid.domain(dims.CellDim)
    cell = domain(h_grid.Zone.HALO)
