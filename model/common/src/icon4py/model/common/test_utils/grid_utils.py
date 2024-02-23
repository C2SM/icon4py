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

from icon4py.model.common.decomposition.definitions import SingleNodeRun
from icon4py.model.common.grid.grid_manager import GridManager, ToGt4PyTransformation
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.test_utils.datatest_utils import (
    SERIALIZED_DATA_PATH,
    create_icon_serial_data_provider,
    get_datapath_for_experiment,
    get_processor_properties_for_run,
    get_ranked_data_path,
    GRIDS_PATH,
    REGIONAL_EXPERIMENT,
    GLOBAL_EXPERIMENT,
)

MCH_CH_R04B09_LEVELS = 65


def get_icon_grid(on_gpu: bool):
    processor_properties = get_processor_properties_for_run(SingleNodeRun())
    ranked_path = get_ranked_data_path(SERIALIZED_DATA_PATH, processor_properties)
    data_path = get_datapath_for_experiment(ranked_path)
    icon_data_provider = create_icon_serial_data_provider(data_path, processor_properties)
    grid_savepoint = icon_data_provider.from_savepoint_grid()
    return grid_savepoint.construct_icon_grid(on_gpu)


@functools.cache
def get_icon_grid_global(on_gpu: bool = False) -> IconGrid:
    return _load_from_file("r02b04_global", "icon_grid_0013_R02B04_R.nc")


@functools.cache
def get_icon_grid_regional(on_gpu: bool = False) -> IconGrid:
    return _load_from_file(REGIONAL_EXPERIMENT, "grid.nc")


def _load_from_file(directory: str, filename: str) -> IconGrid:
    grid_file = GRIDS_PATH.joinpath(directory, filename)
    gm = GridManager(
        ToGt4PyTransformation(), str(grid_file), VerticalGridSize(MCH_CH_R04B09_LEVELS)
    )
    gm()
    return gm.get_grid()


@pytest.fixture
def grid(request):
    return request.param
