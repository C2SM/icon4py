# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.stencils.compute_virtual_potential_temperature import (
    compute_virtual_potential_temperature,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def compute_virtual_potential_temperature_numpy(
    virtual_temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    return virtual_temperature * (constants.P0REF / pressure) ** constants.RD_O_CPD


class TestComputeVirtualPotentialTemperature(stencil_tests.StencilTest):
    PROGRAM = compute_virtual_potential_temperature
    OUTPUTS = ("theta_v",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        virtual_temperature: np.ndarray,
        pressure: np.ndarray,
        **kwargs,
    ) -> dict:
        theta_v = compute_virtual_potential_temperature_numpy(virtual_temperature, pressure)
        return dict(theta_v=theta_v)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        virtual_temperature = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=180.0, high=320.0, dtype=wpfloat
        )
        pressure = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=1.0e3, high=1.05e5, dtype=wpfloat
        )
        theta_v = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            virtual_temperature=virtual_temperature,
            pressure=pressure,
            theta_v=theta_v,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
