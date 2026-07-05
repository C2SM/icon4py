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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_wind_speed import (
    compute_surface_wind_speed,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeSurfaceWindSpeed(StencilTest):
    PROGRAM = compute_surface_wind_speed
    OUTPUTS = ("wind_rel",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        ua: np.ndarray,
        va: np.ndarray,
        reference_u: np.ndarray,
        reference_v: np.ndarray,
        min_sfc_wind: float,
        **kwargs: Any,
    ) -> dict:
        wind_rel = np.maximum(
            min_sfc_wind, np.sqrt((ua - reference_u) ** 2 + (va - reference_v) ** 2)
        )
        return dict(wind_rel=wind_rel)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return dict(
            ua=data_alloc.random_field(grid, dims.CellDim, low=-20.0, high=20.0, dtype=wpfloat),
            va=data_alloc.random_field(grid, dims.CellDim, low=-20.0, high=20.0, dtype=wpfloat),
            reference_u=data_alloc.random_field(
                grid, dims.CellDim, low=-2.0, high=2.0, dtype=wpfloat
            ),
            reference_v=data_alloc.random_field(
                grid, dims.CellDim, low=-2.0, high=2.0, dtype=wpfloat
            ),
            wind_rel=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
            min_sfc_wind=0.3,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
