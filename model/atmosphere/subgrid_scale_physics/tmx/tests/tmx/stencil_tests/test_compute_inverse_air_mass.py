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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_inverse_air_mass import (
    compute_inverse_air_mass,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeInverseAirMass(StencilTest):
    PROGRAM = compute_inverse_air_mass
    OUTPUTS = ("inv_air_mass",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        air_mass: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(inv_air_mass=1.0 / air_mass)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return dict(
            air_mass=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=10.0, high=5.0e3, dtype=wpfloat
            ),
            inv_air_mass=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
