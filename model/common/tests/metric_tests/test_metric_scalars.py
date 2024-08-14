# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.dimension import KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.metrics.metric_scalars import compute_kstart_dd3d
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.common.type_alias import wpfloat


def test_compute_kstart_dd3d():
    grid = SimpleGrid()

    scalfac_dd3d_full = random_field(grid, KDim, low=0.1, high=3.0, dtype=wpfloat).asnumpy()
    scalfac_dd3d_full[0:3] = 0.0
    kstart_dd3d_ref = 3

    kstart_dd3d_full = compute_kstart_dd3d(
        scalfac_dd3d=scalfac_dd3d_full,
    )
    assert kstart_dd3d_ref == kstart_dd3d_full
