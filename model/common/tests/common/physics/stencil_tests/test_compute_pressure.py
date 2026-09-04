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

from icon4py.model.common import constants as phy_const, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics.compute_pressure import (
    compute_surface_and_hydrostatic_pressure,
)
from icon4py.model.testing import stencil_tests


class TestComputeSurfaceAndHydrostaticPressure(stencil_tests.StencilTest):
    PROGRAM = compute_surface_and_hydrostatic_pressure
    OUTPUTS = ("pressure", "pressure_ifc")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        exner: np.ndarray,
        virtual_temperature: np.ndarray,
        ddqz_z_full: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        surface_pressure = phy_const.P0REF * np.exp(
            phy_const.CPD_O_RD * np.log(exner[:, -3])
            + phy_const.GRAV_O_RD
            * (
                ddqz_z_full[:, -1] / virtual_temperature[:, -1]
                + ddqz_z_full[:, -2] / virtual_temperature[:, -2]
                + 0.5 * ddqz_z_full[:, -3] / virtual_temperature[:, -3]
            )
        )
        pressure_ifc = np.zeros_like(virtual_temperature)
        pressure = np.zeros_like(virtual_temperature)
        ground_level = virtual_temperature.shape[1] - 1
        pressure_ifc[:, ground_level] = surface_pressure * np.exp(
            -phy_const.GRAV_O_RD
            * ddqz_z_full[:, ground_level]
            / virtual_temperature[:, ground_level]
        )
        pressure[:, ground_level] = np.sqrt(pressure_ifc[:, ground_level] * surface_pressure)
        for k in range(ground_level - 1, -1, -1):
            pressure_ifc[:, k] = pressure_ifc[:, k + 1] * np.exp(
                -phy_const.GRAV_O_RD * ddqz_z_full[:, k] / virtual_temperature[:, k]
            )
            pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k + 1])

        # the half-level field carries the surface pressure in its bottom entry
        pressure_ifc = np.concatenate([pressure_ifc, surface_pressure[:, None]], axis=1)
        return dict(
            pressure=pressure,
            pressure_ifc=pressure_ifc,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        exner = data_alloc.random_field(dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat)
        virtual_temperature = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=1.0e-2, dtype=ta.wpfloat
        )
        ddqz_z_full = data_alloc.random_field(dims.CellDim, dims.KDim, low=1.0, dtype=ta.wpfloat)
        pressure = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        pressure_ifc = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)

        return dict(
            exner=exner,
            virtual_temperature=virtual_temperature,
            ddqz_z_full=ddqz_z_full,
            pressure=pressure,
            pressure_ifc_on_model_levels=data_alloc.zero_field(
                dims.CellDim, dims.KDim, dtype=ta.wpfloat
            ),
            pressure_ifc=pressure_ifc,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
