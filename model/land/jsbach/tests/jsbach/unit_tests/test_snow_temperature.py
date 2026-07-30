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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.land.jsbach.stencils.snow_temperature import snow_temperature_back_substitution
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager
from icon4py.model.testing.stencil_tests import StencilTest


class TestSnowTemperatureBackSubstitution(StencilTest):
    """Reconstruct the snow temperature column from the lagged R&M coefficients, with
    the variable top snow layer `itop` masking absent (snow-free) upper layers.

    Fortran reference: calc_snow_temperature, mo_sse_process.f90:796-838
        t_snow(:,is)      = t_srf                              (init, all layers)
        t_snow(:,is)      = a(:,is-1) + b(:,is-1)*t_snow(:,is-1)   for is > itop
    Layers with (0-based) index k < itop keep the surface temperature.
    """

    PROGRAM = snow_temperature_back_substitution
    OUTPUTS = ("t_snow",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t_snow_acoef: np.ndarray,
        t_snow_bcoef: np.ndarray,
        t_srf: np.ndarray,
        itop: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        nsnow = t_snow_acoef.shape[1]
        t_snow = np.empty_like(t_snow_acoef)
        t_snow[:] = t_srf[:, np.newaxis]
        for k in range(1, nsnow):
            update = k >= itop
            t_snow[:, k] = np.where(
                update,
                t_snow_acoef[:, k - 1] + t_snow_bcoef[:, k - 1] * t_snow[:, k - 1],
                t_snow[:, k],
            )
        return dict(t_snow=t_snow)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        t_snow_acoef = random_field(grid, dims.CellDim, dims.KDim)
        t_snow_bcoef = random_field(grid, dims.CellDim, dims.KDim)
        t_srf = random_field(grid, dims.CellDim, low=250.0, high=290.0)
        # itop (1-based Fortran top layer) spanning full snow, partial, and snow-free.
        itop_np = np.random.default_rng().integers(
            1, grid.num_levels + 2, size=grid.num_cells, dtype=np.int32
        )
        itop = gtx.as_field((dims.CellDim,), itop_np)
        t_snow = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            t_snow_acoef=t_snow_acoef,
            t_snow_bcoef=t_snow_bcoef,
            t_srf=t_srf,
            itop=itop,
            t_snow=t_snow,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
