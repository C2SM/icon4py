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

from icon4py.model.common.grid.grid_manager import GridManager, ToGt4PyTransformation
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.test_utils.data_handling import download_and_extract
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    GRID_URIS,
    GRIDS_PATH,
    R02B04_GLOBAL,
    REGIONAL_EXPERIMENT,
)


GLOBAL_NUM_LEVELS = 80

MCH_CH_R04B09_LEVELS = 65


@functools.cache
def get_icon_grid_from_gridfile(experiment: str, on_gpu: bool = False) -> IconGrid:
    if experiment == GLOBAL_EXPERIMENT:
        return _download_and_load_from_gridfile(
            R02B04_GLOBAL,
            "icon_grid_0013_R02B04_R.nc",
            num_levels=GLOBAL_NUM_LEVELS,
            on_gpu=on_gpu,
            limited_area=False,
        )
    elif experiment == REGIONAL_EXPERIMENT:
        return _download_and_load_from_gridfile(
            REGIONAL_EXPERIMENT,
            "grid.nc",
            num_levels=MCH_CH_R04B09_LEVELS,
            on_gpu=on_gpu,
            limited_area=True,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def download_grid_file(file_path: str, filename: str):
    grid_file = GRIDS_PATH.joinpath(file_path, filename)
    if not grid_file.exists():
        download_and_extract(
            GRID_URIS[file_path],
            grid_file.parent,
            grid_file.parent,
        )
    return grid_file


def load_grid_from_file(
    grid_file: str, num_levels: int, on_gpu: bool, limited_area: bool
) -> IconGrid:
    gm = GridManager(
        ToGt4PyTransformation(),
        str(grid_file),
        VerticalGridSize(num_levels),
    )
    gm(on_gpu=on_gpu, limited_area=limited_area)
    return gm.get_grid()


def _download_and_load_from_gridfile(
    file_path: str, filename: str, num_levels: int, on_gpu: bool, limited_area: bool
) -> IconGrid:
    grid_file = download_grid_file(file_path, filename)
    return load_grid_from_file(grid_file, num_levels, on_gpu, limited_area)


@pytest.fixture
def grid(request):
    return request.param
