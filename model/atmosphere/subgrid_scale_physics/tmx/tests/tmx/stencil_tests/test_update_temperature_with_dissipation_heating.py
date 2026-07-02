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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.update_temperature_with_dissipation_heating import (
    update_temperature_with_dissipation_heating,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestUpdateTemperatureWithDissipationHeating(StencilTest):
    PROGRAM = update_temperature_with_dissipation_heating
    OUTPUTS = ("dissip_ke", "heating", "new_temperature", "tend_temperature")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u: np.ndarray,
        v: np.ndarray,
        new_u: np.ndarray,
        new_v: np.ndarray,
        air_mass: np.ndarray,
        cv_air: np.ndarray,
        temperature: np.ndarray,
        tend_temperature: np.ndarray,
        q_snocpymlt: np.ndarray,
        dissipation_factor: float,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        dissip_ke = (
            0.5 * air_mass * dissipation_factor / dtime * (u**2 - new_u**2 + v**2 - new_v**2)
        )
        heating = dissip_ke.copy()
        heating[:, -1] -= q_snocpymlt
        tend_temperature = tend_temperature + heating / cv_air
        new_temperature = temperature + tend_temperature * dtime
        return dict(
            dissip_ke=dissip_ke,
            heating=heating,
            new_temperature=new_temperature,
            tend_temperature=tend_temperature,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        def wind() -> gtx.Field:
            return data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=-10.0, high=10.0, dtype=wpfloat
            )

        return dict(
            u=wind(),
            v=wind(),
            new_u=wind(),
            new_v=wind(),
            air_mass=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=100.0, high=1000.0, dtype=wpfloat
            ),
            cv_air=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=700.0, high=800.0, dtype=wpfloat
            ),
            temperature=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=250.0, high=300.0, dtype=wpfloat
            ),
            tend_temperature=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=-1.0e-3, high=1.0e-3, dtype=wpfloat
            ),
            q_snocpymlt=data_alloc.random_field(
                grid, dims.CellDim, low=0.0, high=10.0, dtype=wpfloat
            ),
            dissip_ke=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            heating=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            new_temperature=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            dissipation_factor=wpfloat(1.0),
            dtime=wpfloat(300.0),
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
