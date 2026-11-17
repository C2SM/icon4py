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

from icon4py.model.common import field
from icon4py.model.common.framework import quantities as qty
from icon4py.model.common.grid import base, simple
from icon4py.model.common.io import utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils


# setting backend to fieldview embedded here.
backend = None
UNLIMITED = None

grid_file = dt_utils.get_grid_filepath(test_defs.Grids.R02B04_GLOBAL)
global_grid = grid_utils.get_grid_manager_from_identifier(
    test_defs.Experiments.EXCLAIM_APE.grid,
    num_levels=60,
    keep_skip_values=True,
    allocator=backend,  # type: ignore[arg-type]  # None selects the embedded backend
).grid


def model_state(grid: base.Grid) -> dict[str, field.Field]:
    """The prognostic variables, random, by standard name."""
    return {
        quantity.standard_name: field.Field(
            quantity, data_alloc.random_field(grid, *quantity.dims, dtype=np.float32)
        )
        for quantity in (
            qty.RhoOnCellK,
            qty.ExnerOnCellK,
            qty.ThetaVOnCellK,
            qty.WOnCellKHalf,
            qty.VnOnEdgeK,
        )
        if quantity.standard_name is not None
    }


def model_dataarrays(grid: base.Grid) -> dict[str, xr.DataArray]:
    """`model_state` as the writers take it."""
    return {name: utils.to_data_array(field_) for name, field_ in model_state(grid).items()}


def state_values() -> Iterator[xr.DataArray]:
    yield from model_dataarrays(simple.simple_grid()).values()
