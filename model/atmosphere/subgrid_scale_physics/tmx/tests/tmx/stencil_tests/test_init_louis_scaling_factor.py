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
    init_louis_scaling_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestInitLouisScalingFactor(stencil_tests.StencilTest):
    PROGRAM = init_louis_scaling_factor
    OUTPUTS = ("scaling_factor_louis",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        cell_area: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(scaling_factor_louis=97294071.23714285 / cell_area)  # mean_cell_area_r2b8

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        cell_area = data_alloc.random_field(dims.CellDim, low=1.0e6, high=1.0e8, dtype=wpfloat)
        scaling_factor_louis = data_alloc.zero_field(dims.CellDim, dtype=wpfloat)

        return dict(
            cell_area=cell_area,
            scaling_factor_louis=scaling_factor_louis,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
