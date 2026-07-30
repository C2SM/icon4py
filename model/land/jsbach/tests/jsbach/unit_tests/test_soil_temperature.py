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
from icon4py.model.land.jsbach.stencils.soil_temperature import soil_temperature_back_substitution
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager
from icon4py.model.testing.stencil_tests import StencilTest


class TestSoilTemperatureBackSubstitution(StencilTest):
    """Reconstruct the soil temperature column from the (lagged) Richtmyer-Morton
    coefficients: the back-substitution half of the JSBACH soil energy solve.

    Fortran reference: calc_soil_temperature, mo_sse_process.f90:487-504
        t_soil_sl(:,1)    = t_soil_top
        t_soil_sl(:,k+1)  = t_soil_acoef(:,k) + t_soil_bcoef(:,k) * t_soil_sl(:,k)
    """

    PROGRAM = soil_temperature_back_substitution
    OUTPUTS = ("soil_temperature",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        acoef: np.ndarray,
        bcoef: np.ndarray,
        surface_temperature: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        nsoil = acoef.shape[1]
        soil_temperature = np.zeros_like(acoef)
        soil_temperature[:, 0] = surface_temperature
        for k in range(1, nsoil):
            soil_temperature[:, k] = acoef[:, k - 1] + bcoef[:, k - 1] * soil_temperature[:, k - 1]
        return dict(soil_temperature=soil_temperature)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        # bcoef in [0, 1) keeps the recurrence stable (a physical R&M b-coefficient
        # is a bounded weight); acoef is an unconstrained temperature-like offset.
        acoef = random_field(grid, dims.CellDim, dims.KDim)
        bcoef = random_field(grid, dims.CellDim, dims.KDim)
        surface_temperature = random_field(grid, dims.CellDim)
        soil_temperature = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            acoef=acoef,
            bcoef=bcoef,
            surface_temperature=surface_temperature,
            soil_temperature=soil_temperature,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
