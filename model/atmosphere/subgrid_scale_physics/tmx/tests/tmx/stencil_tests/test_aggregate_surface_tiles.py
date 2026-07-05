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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.aggregate_surface_tiles import (
    aggregate_surface_tiles,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestAggregateSurfaceTiles(StencilTest):
    PROGRAM = aggregate_surface_tiles
    OUTPUTS = ("grid_mean",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        field_ocean: np.ndarray,
        field_ice: np.ndarray,
        field_land: np.ndarray,
        fraction_ocean: np.ndarray,
        fraction_ice: np.ndarray,
        fraction_land: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        grid_mean = (
            np.where(fraction_ocean > 0.0, fraction_ocean * field_ocean, 0.0)
            + np.where(fraction_ice > 0.0, fraction_ice * field_ice, 0.0)
            + np.where(fraction_land > 0.0, fraction_land * field_land, 0.0)
        )
        return dict(grid_mean=grid_mean)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            field_ocean=data_alloc.random_field(grid, cell, low=-100.0, high=100.0, dtype=wpfloat),
            field_ice=data_alloc.random_field(grid, cell, low=-100.0, high=100.0, dtype=wpfloat),
            field_land=data_alloc.random_field(grid, cell, low=-100.0, high=100.0, dtype=wpfloat),
            fraction_ocean=data_alloc.random_field(grid, cell, low=0.0, high=1.0, dtype=wpfloat),
            fraction_ice=data_alloc.random_field(grid, cell, low=0.0, high=1.0, dtype=wpfloat),
            fraction_land=data_alloc.random_field(grid, cell, low=0.0, high=1.0, dtype=wpfloat),
            grid_mean=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
