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
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    GRIDS_PATH,
    REGIONAL_EXPERIMENT,
)


MCH_CH_R04B09_LEVELS = 65


@functools.cache
def get_icon_grid_from_gridfile(experiment: str, on_gpu: bool = False) -> IconGrid:
    if experiment == GLOBAL_EXPERIMENT:
        return _load_from_gridfile("r02b04_global", "icon_grid_0013_R02B04_R.nc", on_gpu=on_gpu)
    elif experiment == REGIONAL_EXPERIMENT:
        return _load_from_gridfile(REGIONAL_EXPERIMENT, "grid.nc", on_gpu=on_gpu)
    else:
        raise ValueError(f"Unknown grid file for: {experiment}")


def _load_from_gridfile(directory: str, filename: str, on_gpu: bool) -> IconGrid:
    grid_file = GRIDS_PATH.joinpath(directory, filename)
    gm = GridManager(
        ToGt4PyTransformation(),
        str(grid_file),
        VerticalGridSize(MCH_CH_R04B09_LEVELS),
    )
    gm(on_gpu=on_gpu)
    return gm.get_grid()


@pytest.fixture
def grid(request):
    return request.param
