# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime

import numpy as np

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.diagnostic_calculations import diagnostic_components
from icon4py.model.common.grid import simple, vertical as v_grid
from icon4py.model.common.states import data, model
from icon4py.model.common.test_utils import helpers


def test_diagnostic_component(backend):
    run_time = backend
    grid = simple.SimpleGrid()
    dummy_vertical_vector = helpers.zero_field(grid, dims.KDim)
    vertical_grid = v_grid.VerticalGrid(
        v_grid.VerticalGridConfig(grid.num_levels), dummy_vertical_vector, dummy_vertical_vector
    )
    qv = helpers.constant_field(grid, 1.0, dims.CellDim, dims.KDim)
    qc = helpers.constant_field(grid, 0.1, dims.CellDim, dims.KDim)
    qi = helpers.constant_field(grid, 0.1, dims.CellDim, dims.KDim)
    qr = helpers.constant_field(grid, 0.1, dims.CellDim, dims.KDim)
    qs = helpers.constant_field(grid, 0.1, dims.CellDim, dims.KDim)
    qg = helpers.constant_field(grid, 0.1, dims.CellDim, dims.KDim)
    theta_v = helpers.constant_field(grid, 2.0, dims.CellDim, dims.KDim, dtype=float)
    exner = helpers.constant_field(grid, 8.0, dims.CellDim, dims.KDim, dtype=float)
    state = {
        "theta_v": model.ModelField(
            theta_v, data.PROGNOSTIC_CF_ATTRIBUTES["virtual_potential_temperature"]
        ),
        "dimensionless_exner_pressure": model.ModelField(
            exner, data.PROGNOSTIC_CF_ATTRIBUTES["exner_function"]
        ),
        "specific_humidity": model.ModelField(
            qv, data.COMMON_TRACER_CF_ATTRIBUTES["specific_humidity"]
        ),
        "specific_rain_content": model.ModelField(
            qr, data.COMMON_TRACER_CF_ATTRIBUTES["specific_rain"]
        ),
        "specific_snow_content": model.ModelField(
            qs, data.COMMON_TRACER_CF_ATTRIBUTES["specific_snow"]
        ),
        "specific_graupel_content": model.ModelField(
            qg, data.COMMON_TRACER_CF_ATTRIBUTES["specific_graupel"]
        ),
        "specific_ice_content": model.ModelField(
            qi, data.COMMON_TRACER_CF_ATTRIBUTES["specific_ice"]
        ),
        "specific_cloud_content": model.ModelField(
            qc, data.COMMON_TRACER_CF_ATTRIBUTES["specific_cloud"]
        ),
    }

    comp = diagnostic_components.TemperatureComponent(
        backend=run_time, grid=grid, vertical_grid=vertical_grid
    )

    temp_diagnostics = comp(state, datetime.datetime.now())
    assert np.all(temp_diagnostics["virtual_temperature"].data.asnumpy() == 16.0)
    assert np.all(
        temp_diagnostics["temperature"].data.asnumpy()
        == 16.0 / (constants.GAS_CONSTANT_WATER_VAPOR / constants.GAS_CONSTANT_DRY_AIR - 0.5)
    )
