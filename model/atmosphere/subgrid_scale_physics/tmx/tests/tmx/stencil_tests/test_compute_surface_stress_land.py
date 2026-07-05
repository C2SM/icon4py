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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_stress_land import (
    compute_surface_stress_land,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeSurfaceStressLand(StencilTest):
    PROGRAM = compute_surface_stress_land
    OUTPUTS = ("u_stress", "v_stress")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_sfc: np.ndarray,
        km: np.ndarray,
        wind_rel: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(
            u_stress=rho_sfc * km * wind_rel * ua,
            v_stress=rho_sfc * km * wind_rel * va,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            rho_sfc=data_alloc.random_field(grid, cell, low=1.0, high=1.4, dtype=wpfloat),
            km=data_alloc.random_field(grid, cell, low=1.0e-4, high=0.05, dtype=wpfloat),
            wind_rel=data_alloc.random_field(grid, cell, low=0.3, high=20.0, dtype=wpfloat),
            ua=data_alloc.random_field(grid, cell, low=-20.0, high=20.0, dtype=wpfloat),
            va=data_alloc.random_field(grid, cell, low=-20.0, high=20.0, dtype=wpfloat),
            u_stress=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            v_stress=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
