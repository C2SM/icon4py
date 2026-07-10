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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.init_height_above_ground import (
    init_height_above_ground,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def init_height_above_ground_numpy(
    z_mc: np.ndarray,
    z_ifc_sfc: np.ndarray,
) -> np.ndarray:
    return z_mc - z_ifc_sfc[:, np.newaxis]


class TestInitHeightAboveGround(StencilTest):
    PROGRAM = init_height_above_ground
    OUTPUTS = ("height_above_ground",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_mc: np.ndarray,
        z_ifc_sfc: np.ndarray,
        **kwargs,
    ) -> dict:
        height_above_ground = init_height_above_ground_numpy(z_mc, z_ifc_sfc)
        return dict(height_above_ground=height_above_ground)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_mc = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        )
        z_ifc_sfc = data_alloc.random_field(grid, dims.CellDim, low=0.0, high=3.0e3, dtype=wpfloat)
        height_above_ground = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            z_mc=z_mc,
            z_ifc_sfc=z_ifc_sfc,
            height_above_ground=height_above_ground,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
