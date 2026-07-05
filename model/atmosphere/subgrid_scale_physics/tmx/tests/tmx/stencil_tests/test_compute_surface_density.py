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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_density import (
    compute_surface_density,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeSurfaceDensity(StencilTest):
    PROGRAM = compute_surface_density
    OUTPUTS = ("rho_sfc",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        surface_pressure: np.ndarray,
        temperature_sfc: np.ndarray,
        qsat_sfc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        rho_sfc = surface_pressure / (
            constants.RD * temperature_sfc * (1.0 + constants.RV_O_RD_MINUS_1 * qsat_sfc)
        )
        return dict(rho_sfc=rho_sfc)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return dict(
            surface_pressure=data_alloc.random_field(
                grid, dims.CellDim, low=9.0e4, high=1.05e5, dtype=wpfloat
            ),
            temperature_sfc=data_alloc.random_field(
                grid, dims.CellDim, low=240.0, high=310.0, dtype=wpfloat
            ),
            qsat_sfc=data_alloc.random_field(grid, dims.CellDim, low=0.0, high=0.03, dtype=wpfloat),
            rho_sfc=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
