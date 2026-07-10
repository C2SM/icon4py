# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.init_louis_scaling_factor import (
    MEAN_CELL_AREA_R2B8,
    init_louis_scaling_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestInitLouisScalingFactor(StencilTest):
    PROGRAM = init_louis_scaling_factor
    OUTPUTS = ("scaling_factor_louis",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        cell_area: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(scaling_factor_louis=MEAN_CELL_AREA_R2B8 / cell_area)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        cell_area = data_alloc.random_field(
            grid, dims.CellDim, low=1.0e6, high=1.0e8, dtype=wpfloat
        )
        scaling_factor_louis = data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat)

        return dict(
            cell_area=cell_area,
            scaling_factor_louis=scaling_factor_louis,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
