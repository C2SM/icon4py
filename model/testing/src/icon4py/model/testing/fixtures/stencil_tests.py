# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

import pytest
from gt4py.next import backend as gtx_backend

from icon4py.model.common.grid import base as base_grid, grid_manager as gm, simple as simple_grid
from icon4py.model.testing import definitions


DEFAULT_GRID: Final[str] = "simple_grid"
VALID_GRIDS: tuple[str, str, str] = ("simple_grid", "icon_grid", "icon_grid_global")


def _check_grid_validity(grid_name: str) -> None:
    assert (
        grid_name in VALID_GRIDS
    ), f"Invalid value for '--grid' option - possible names are {VALID_GRIDS}"


def _get_grid_manager(
    descriptor: definitions.Grid, num_levels: int, backend: gtx_backend.Backend | None
) -> gm.GridManager:
    from icon4py.model.testing.grid_utils import get_grid_manager

    assert descriptor.file_name is not None
    return get_grid_manager(
        grid_file=descriptor.file_name,
        num_levels=num_levels,
        keep_skip_values=False,
        backend=backend,
    )


def _get_grid(
    selected_grid_type: str, selected_backend: gtx_backend.Backend | None
) -> base_grid.Grid:
    print(f"Using grid type: {selected_grid_type} with backend: {selected_backend}")
    print("This should have been called only once per test session.")
    match selected_grid_type:
        case "icon_grid":
            return _get_grid_manager(
                descriptor=definitions.Grids.MCH_CH_R04B09_DSL,
                num_levels=65,  # random decision
                backend=selected_backend,
            ).grid
        case "icon_grid_global":
            return _get_grid_manager(
                descriptor=definitions.Grids.R02B04_GLOBAL,
                num_levels=60,  # random decision
                backend=selected_backend,
            ).grid
        case _:
            return simple_grid.simple_grid(selected_backend)


@pytest.fixture(scope="session")
def grid(request: pytest.FixtureRequest, backend: gtx_backend.Backend | None) -> base_grid.Grid:
    try:
        grid_option = request.config.getoption("grid")
    except ValueError:
        grid_option = DEFAULT_GRID
    else:
        _check_grid_validity(grid_option)
    grid = _get_grid(grid_option, backend)
    return grid
