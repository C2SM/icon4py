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
from icon4py.model.testing import definitions, grid_utils


DEFAULT_GRID: Final[str] = "simple"
DEFAULT_NUM_LEVELS: Final[int] = (
    10  # the number matters for performance measurements, but otherwise is arbitrary
)
VALID_GRID_PRESETS: tuple[str, ...] = (
    "simple",
    "icon_regional",
    "icon_global",
    "icon_benchmark",
)


def _get_grid_from_preset(
    grid_preset: str,
    *,
    num_levels: int = DEFAULT_NUM_LEVELS,
    backend: gtx_backend.Backend | None = None,
) -> base_grid.Grid:
    match grid_preset:
        case "icon_regional":
            return grid_utils.get_grid_manager_from_identifier(
                definitions.Grids.MCH_CH_R04B09_DSL,
                num_levels=num_levels,
                keep_skip_values=False,
                backend=backend,
            ).grid
        case "icon_global":
            return grid_utils.get_grid_manager_from_identifier(
                definitions.Grids.R02B04_GLOBAL,
                num_levels=num_levels,
                keep_skip_values=False,
                backend=backend,
            ).grid
        case "icon_benchmark":
            return grid_utils.get_grid_manager_from_identifier(
                definitions.Grids.MCH_OPR_R19B08_DOMAIN01,
                num_levels=80,  # default benchmark size in ICON Fortran
                keep_skip_values=False,
                backend=backend,
            ).grid
        case _:
            return simple_grid.simple_grid(backend=backend, num_levels=num_levels)


@pytest.fixture(scope="session")
def grid(request: pytest.FixtureRequest, backend: gtx_backend.Backend | None) -> base_grid.Grid:
    """
    Fixture for providing a grid instance.

    The provided grid instance is based on the configuration specified in the
    pytest command line option `--grid <grid_name>:<grid_levels>`, where `<grid_name>`
    might refer to a known grid configuration or to an existing ICON NetCDF grid file,
    and `<grid_levels>` specifies the number of vertical levels to use (optional).
    """
    spec = request.config.getoption("grid")
    if spec is None:
        spec = DEFAULT_GRID
    assert isinstance(spec, str), "Grid spec must be a string"
    if spec.count(":") > 1:
        raise ValueError("Invalid grid spec in '--grid' option (spec: <grid_name>:<grid_levels>)")

    name, *levels = spec.split(":")
    num_levels = int(levels[0]) if levels and levels[0].strip() else DEFAULT_NUM_LEVELS

    if name in VALID_GRID_PRESETS:
        grid = _get_grid_from_preset(name, num_levels=num_levels, backend=backend)
    else:
        try:
            grid_file = pathlib.Path(name).resolve(strict=True)
            grid = grid_utils.get_grid_manager(
                grid_file, num_levels=num_levels, keep_skip_values=False, backend=backend
            ).grid
        except OSError as e:
            raise ValueError(
                f"Invalid grid name in '--grid' option. It should be one of {VALID_GRID_PRESETS}"
                " or a valid path to an ICON NetCDF grid file."
            ) from e

    return grid
