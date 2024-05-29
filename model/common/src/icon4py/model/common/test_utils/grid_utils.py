# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import functools

import pytest

import icon4py.model.common.grid.grid_manager as gm
import icon4py.model.common.test_utils.datatest_utils as dt_utils
from icon4py.model.common.grid import icon, vertical


REGIONAL_GRIDFILE = "grid.nc"

GLOBAL_GRIDFILE = "icon_grid_0013_R02B04_R.nc"

GLOBAL_NUM_LEVELS = 80

MCH_CH_R04B09_LEVELS = 65


@functools.cache
def get_icon_grid_from_gridfile(experiment: str, on_gpu: bool = False) -> icon.IconGrid:
    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        return _load_from_gridfile(
            dt_utils.R02B04_GLOBAL,
            GLOBAL_GRIDFILE,
            num_levels=GLOBAL_NUM_LEVELS,
            on_gpu=on_gpu,
            limited_area=False,
        )
    elif experiment == dt_utils.REGIONAL_EXPERIMENT:
        return _load_from_gridfile(
            dt_utils.REGIONAL_EXPERIMENT,
            REGIONAL_GRIDFILE,
            num_levels=MCH_CH_R04B09_LEVELS,
            on_gpu=on_gpu,
            limited_area=True,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def _load_from_gridfile(
    file_path: str, filename: str, num_levels: int, on_gpu: bool, limited_area: bool
) -> icon.IconGrid:
    grid_file = dt_utils.GRIDS_PATH.joinpath(file_path, filename)
    if not grid_file.exists():
        from icon4py.model.common.test_utils.data_handling import download_and_extract

        download_and_extract(
            dt_utils.GRID_URIS[file_path],
            grid_file.parent,
            grid_file.parent,
        )
    manager = gm.GridManager(
        gm.ToGt4PyTransformation(),
        str(grid_file),
        vertical.VerticalGridSize(num_levels),
    )
    manager(on_gpu=on_gpu, limited_area=limited_area)
    return manager.grid


@pytest.fixture
def grid(request):
    return request.param
