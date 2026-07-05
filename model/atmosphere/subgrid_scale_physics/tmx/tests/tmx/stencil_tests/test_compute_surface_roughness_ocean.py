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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_roughness_ocean import (
    compute_surface_roughness_ocean,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeSurfaceRoughnessOcean(StencilTest):
    PROGRAM = compute_surface_roughness_ocean
    OUTPUTS = ("rough_m", "rough_h")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wind_rel: np.ndarray,
        km: np.ndarray,
        charnock: float,
        viscous_coeff: float,
        kinematic_viscosity: float,
        z0m_min: float,
        **kwargs: Any,
    ) -> dict:
        rough = wind_rel**2 * km * charnock / constants.GRAV + viscous_coeff * np.minimum(
            0.01, kinematic_viscosity / (np.sqrt(km) * wind_rel)
        )
        rough = np.maximum(z0m_min, rough)
        return dict(rough_m=rough, rough_h=rough)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return dict(
            wind_rel=data_alloc.random_field(grid, dims.CellDim, low=0.3, high=20.0, dtype=wpfloat),
            km=data_alloc.random_field(grid, dims.CellDim, low=1.0e-4, high=0.1, dtype=wpfloat),
            rough_m=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
            rough_h=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
            charnock=0.018,
            viscous_coeff=0.11,
            kinematic_viscosity=1.5e-5,
            z0m_min=1.5e-5,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
