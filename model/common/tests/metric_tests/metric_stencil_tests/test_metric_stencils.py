# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.metrics.metric_fields import compute_z_mc
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers as testing_helpers


class TestComputeZMc(testing_helpers.StencilTest):
    PROGRAM = compute_z_mc
    OUTPUTS = ("z_mc",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        z_ifc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        shp = z_ifc.shape
        z_mc = 0.5 * (z_ifc + np.roll(z_ifc, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(z_mc=z_mc)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict:
        z_mc = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_if = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        horizontal_start = 0
        horizontal_end = grid.num_cells
        vertical_start = 0
        vertical_end = grid.num_levels

        return dict(
            z_mc=z_mc,
            z_ifc=z_if,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
