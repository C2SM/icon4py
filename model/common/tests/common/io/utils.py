# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import xarray as xr

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, simple
from icon4py.model.common.io import utils
from icon4py.model.common.states import data
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils, grid_utils


# setting backend to fieldview embedded here.
backend = None
UNLIMITED = None
simple_grid = simple.simple_grid()

grid_file = datatest_utils.GRIDS_PATH.joinpath(
    datatest_utils.R02B04_GLOBAL, grid_utils.GLOBAL_GRIDFILE
)
global_grid = grid_utils.get_grid_manager_for_experiment(
    datatest_utils.GLOBAL_EXPERIMENT, keep_skip_values=True, backend=backend
).grid


def model_state(grid: base.Grid) -> dict[str, xr.DataArray]:
    rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    w = data_alloc.random_field(
        grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=np.float32
    )
    vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=np.float32)
    return {
        "air_density": utils.to_data_array(rho, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"]),
        "exner_function": utils.to_data_array(
            exner, data.PROGNOSTIC_CF_ATTRIBUTES["exner_function"]
        ),
        "theta_v": utils.to_data_array(
            theta_v,
            data.PROGNOSTIC_CF_ATTRIBUTES["virtual_potential_temperature"],
            is_on_interface=False,
        ),
        "upward_air_velocity": utils.to_data_array(
            w,
            data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"],
            is_on_interface=True,
        ),
        "normal_velocity": utils.to_data_array(
            vn, data.PROGNOSTIC_CF_ATTRIBUTES["normal_velocity"], is_on_interface=False
        ),
    }


def state_values() -> xr.DataArray:
    state = model_state(simple_grid)
    for v in state.values():
        yield v
