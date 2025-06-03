# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import uuid

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.common.grid import base, gridfile, icon, simple
from icon4py.model.common.grid.base import HorizontalGridSize
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils

from . import utils


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
@pytest.mark.parametrize("keep_skip_values", (True, False))
def test_has_skip_values(grid_file, keep_skip_values):
    grid = utils.run_grid_manager(grid_file, keep_skip_values=keep_skip_values, backend=None).grid
    assert keep_skip_values == grid.has_skip_values()


@pytest.mark.parametrize("limited_area", (True, False))
@pytest.mark.parametrize("keep_skip_values", (True, False))
def test_has_skip_values_on_torus(
    limited_area,
    keep_skip_values,
):
    config = base.GridConfig(
        horizontal_config=HorizontalGridSize(3, 1, 1),
        vertical_size=1,
        keep_skip_values=keep_skip_values,
        limited_area=limited_area,
    )
    grid = (
        icon.IconGrid(uuid.uuid4())
        .set_config(config)
        .set_global_params(
            icon.GlobalGridParams(root=0, level=2, geometry_type=base.GeometryType.TORUS)
        )
    )

    assert (limited_area and keep_skip_values) == grid.has_skip_values()


@pytest.mark.parametrize("limited_area", (True, False))
@pytest.mark.parametrize("keep_skip_values", (True, False))
def test_has_skip_values_on_icosahedron_returns_keep_skip_value(keep_skip_values, limited_area):
    config = base.GridConfig(
        horizontal_config=HorizontalGridSize(3, 1, 1),
        vertical_size=1,
        keep_skip_values=keep_skip_values,
        limited_area=limited_area,
    )
    grid = (
        icon.IconGrid(uuid.uuid4())
        .set_config(config)
        .set_global_params(
            icon.GlobalGridParams(root=0, level=2, geometry_type=base.GeometryType.ICOSAHEDRON)
        )
    )

    assert keep_skip_values == grid.has_skip_values()


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
def test_replace_skip_values(grid_file, caplog, backend):
    caplog.set_level(logging.DEBUG)
    xp = data_alloc.import_array_ns(backend)
    clear_skip_values = functools.partial(base.replace_skip_values, array_ns=xp)

    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=None).grid
    horizontal_connectivities = (
        c for c in grid.neighbor_tables.values() if isinstance(c, gtx.Connectivity)
    )
    for connectivity in horizontal_connectivities:
        clear_skip_values(connectivity)
        assert not np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item()


def test_replace_skip_values_validate_disconnected_grids():
    grid = simple.SimpleGrid()
    connectivity = grid.get_connectivity("V2E")
    connectivity.ndarray[2, :] = gridfile.GridFile.INVALID_INDEX

    with pytest.raises(AssertionError) as error:
        base.replace_skip_values(connectivity)
        assert "disconnected" in error.value
