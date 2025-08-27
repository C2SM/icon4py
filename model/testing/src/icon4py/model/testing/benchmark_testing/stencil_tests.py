# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Final

import pytest
from gt4py.next import backend as gtx_backend

from icon4py.model.common.grid import base as base_grid, simple as simple_grid
from icon4py.model.common.grid.base import Grid
from icon4py.model.common.grid.grid_manager import GridManager
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils
from icon4py.model.testing.pytest_hooks import parse_grid_spec


DEFAULT_GRID: Final[str] = "simple"
DEFAULT_NUM_LEVELS: Final[int] = (
    10  # the number matters for performance measurements, but otherwise is arbitrary
)
VALID_GRID_PRESETS: tuple[str, str, str] = ("simple", "icon_regional", "icon_global")


def _get_grid_from_preset(
    grid_preset: str,
    *,
    num_levels: int = DEFAULT_NUM_LEVELS,
    backend: gtx_backend.Backend | None = None,
) -> Grid | GridManager:
    match grid_preset:
        case "icon_regional":
            return grid_utils.get_grid_manager_from_identifier(
                dt_utils.REGIONAL_EXPERIMENT,
                num_levels=num_levels,
                keep_skip_values=False,
                backend=backend,
            )
        case "icon_global":
            return grid_utils.get_grid_manager_from_identifier(
                dt_utils.R02B04_GLOBAL,
                num_levels=num_levels,
                keep_skip_values=False,
                backend=backend,
            )
        case _:
            return simple_grid.simple_grid(backend=backend, num_levels=num_levels)


@pytest.fixture(scope="session")
def grid_manager(request: pytest.FixtureRequest, backend: gtx_backend.Backend | None) -> Grid | GridManager:
    """
    Fixture for providing a grid instance.

    The provided grid instance is based on the configuration specified in the
    pytest command line option `--grid <grid_name>:<grid_levels>`, where `<grid_name>`
    might refer to a known grid configuration or to an existing ICON NetCDF grid file,
    and `<grid_levels>` specifies the number of vertical levels to use (optional).
    """
    name, num_levels = parse_grid_spec(request.config.getoption("grid"))

    if name in VALID_GRID_PRESETS:
        grid_manager = _get_grid_from_preset(name, num_levels=num_levels, backend=backend)
    else:
        try:
            grid_file = pathlib.Path(name).resolve(strict=True)
            grid_manager = grid_utils.get_grid_manager(
                grid_file, num_levels=num_levels, keep_skip_values=False, backend=backend
            )
        except OSError as e:
            raise ValueError(
                f"Invalid grid name in '--grid' option. It should be one of {VALID_GRID_PRESETS}"
                " or a valid path to an ICON NetCDF grid file."
            ) from e

    return grid_manager
