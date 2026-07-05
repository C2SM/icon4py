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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_potential_temperatures import (
    compute_potential_temperatures,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputePotentialTemperatures(StencilTest):
    PROGRAM = compute_potential_temperatures
    OUTPUTS = ("theta_atm", "theta_sfc")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        temperature_atm: np.ndarray,
        pressure_atm: np.ndarray,
        temperature_sfc: np.ndarray,
        surface_pressure: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        exponent = constants.RD_O_CPD
        theta_atm = temperature_atm * (constants.P0REF / pressure_atm) ** exponent
        theta_sfc = temperature_sfc * (constants.P0REF / surface_pressure) ** exponent
        return dict(theta_atm=theta_atm, theta_sfc=theta_sfc)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return dict(
            temperature_atm=data_alloc.random_field(
                grid, dims.CellDim, low=240.0, high=310.0, dtype=wpfloat
            ),
            pressure_atm=data_alloc.random_field(
                grid, dims.CellDim, low=8.0e4, high=1.0e5, dtype=wpfloat
            ),
            temperature_sfc=data_alloc.random_field(
                grid, dims.CellDim, low=240.0, high=310.0, dtype=wpfloat
            ),
            surface_pressure=data_alloc.random_field(
                grid, dims.CellDim, low=9.0e4, high=1.05e5, dtype=wpfloat
            ),
            theta_atm=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
            theta_sfc=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
