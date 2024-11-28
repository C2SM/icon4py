# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next.backend as gtx_backend
import pytest

import icon4py.model.common.grid.grid_manager as gm
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils import data_handling, datatest_utils as dt_utils


REGIONAL_GRIDFILE = "grid.nc"

GLOBAL_GRIDFILE = "icon_grid_0013_R02B04_R.nc"

GLOBAL_NUM_LEVELS = 60

MCH_CH_R04B09_LEVELS = 65


def get_icon_grid_from_gridfile(
    experiment: str, backend: gtx_backend.Backend = None
) -> gm.GridManager:
    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        return _download_and_load_from_gridfile(
            dt_utils.R02B04_GLOBAL,
            GLOBAL_GRIDFILE,
            num_levels=GLOBAL_NUM_LEVELS,
            limited_area=False,
            backend=backend,
        )
    elif experiment == dt_utils.REGIONAL_EXPERIMENT:
        return _download_and_load_from_gridfile(
            dt_utils.REGIONAL_EXPERIMENT,
            REGIONAL_GRIDFILE,
            num_levels=MCH_CH_R04B09_LEVELS,
            limited_area=True,
            backend=backend,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def download_grid_file(file_path: str, filename: str):
    grid_file = dt_utils.GRIDS_PATH.joinpath(file_path, filename)
    if not grid_file.exists():
        data_handling.download_and_extract(
            dt_utils.GRID_URIS[file_path],
            grid_file.parent,
            grid_file.parent,
        )
    return grid_file


def load_grid_from_file(
    grid_file: str, num_levels: int, backend: gtx_backend.Backend, limited_area: bool
) -> gm.GridManager:
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(),
        str(grid_file),
        v_grid.VerticalGridConfig(num_levels=num_levels),
    )
    manager(backend=backend, limited_area=limited_area)
    return manager


def _download_and_load_from_gridfile(
    file_path: str, filename: str, num_levels: int, backend: gtx_backend.Backend, limited_area: bool
) -> gm.GridManager:
    grid_file = download_grid_file(file_path, filename)

    gm = load_grid_from_file(grid_file, num_levels, backend, limited_area)
    return gm


@pytest.fixture
def grid(request):
    return request.param
