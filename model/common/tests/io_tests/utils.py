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
from icon4py.model.common.grid import base
from icon4py.model.common.io import utils
from icon4py.model.common.states import data
from icon4py.model.common.utils import data_allocation as data_alloc


def model_state(grid: base.BaseGrid) -> dict[str, xr.DataArray]:
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
