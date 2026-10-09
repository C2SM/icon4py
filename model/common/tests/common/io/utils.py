# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterator

import numpy as np
import xarray as xr

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, simple
from icon4py.model.common.io import utils
from icon4py.model.common.states import model
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils


# setting backend to fieldview embedded here.
backend = None
UNLIMITED = None
simple_grid = simple.simple_grid()

grid_file = dt_utils.get_grid_filepath(test_defs.Grids.R02B04_GLOBAL)
global_grid = grid_utils.get_grid_manager_from_identifier(
    test_defs.Experiments.EXCLAIM_APE.grid,
    num_levels=60,
    keep_skip_values=True,
    allocator=backend,  # type: ignore[arg-type]  # None selects the embedded backend
).grid


# CF attributes of the prognostic variables, keyed by output variable name
CF_ATTRIBUTES: dict[str, model.FieldMetaData] = {
    "air_density": model.FieldMetaData(
        standard_name="air_density", long_name="density", units="kg m-3", icon_var_name="rho"
    ),
    "exner_function": model.FieldMetaData(
        standard_name="dimensionless_exner_function",
        long_name="exner function",
        units="1",
        icon_var_name="exner",
    ),
    "theta_v": model.FieldMetaData(
        standard_name="virtual_potential_temperature",
        long_name="virtual potential temperature",
        units="K",
        icon_var_name="theta_v",
    ),
    "upward_air_velocity": model.FieldMetaData(
        standard_name="upward_air_velocity",
        long_name="vertical air velocity component",
        units="m s-1",
        icon_var_name="w",
    ),
    "normal_velocity": model.FieldMetaData(
        standard_name="normal_velocity",
        long_name="velocity normal to edge",
        units="m s-1",
        icon_var_name="vn",
    ),
}


def model_state(grid: base.Grid) -> dict[str, xr.DataArray]:
    rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    w = data_alloc.random_field(grid, dims.CellDim, dims.KHalfDim, dtype=np.float32)
    vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=np.float32)
    return {
        "air_density": utils.to_data_array(rho, CF_ATTRIBUTES["air_density"]),
        "exner_function": utils.to_data_array(exner, CF_ATTRIBUTES["exner_function"]),
        "theta_v": utils.to_data_array(
            theta_v,
            CF_ATTRIBUTES["theta_v"],
        ),
        "upward_air_velocity": utils.to_data_array(
            w,
            CF_ATTRIBUTES["upward_air_velocity"],
        ),
        "normal_velocity": utils.to_data_array(vn, CF_ATTRIBUTES["normal_velocity"]),
    }


def state_values() -> Iterator[xr.DataArray]:
    state = model_state(simple_grid)
    yield from state.values()
