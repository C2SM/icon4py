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
import gt4py.next as gtx
from gt4py.next import backend as gtx_backend
from typing import Dict
import icon4py.model.common.dimension as dims
from icon4py.model.common.grid import base as base_grid, simple as simple_grid
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils
from icon4py.model.testing.pytest_hooks import parse_grid_spec
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import icon
from icon4py.model.common.utils import device_utils
from icon4py.model.common.utils import data_allocation as data_alloc

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
                dt_utils.REGIONAL_EXPERIMENT,
                num_levels=num_levels,
                keep_skip_values=False,
                backend=backend,
            ).grid
        case "icon_global":
            return grid_utils.get_grid_manager_from_identifier(
                dt_utils.R02B04_GLOBAL,
                num_levels=num_levels,
                keep_skip_values=False,
                backend=backend,
            ).grid
        case "icon_benchmark":
            return grid_utils.get_grid_manager_from_identifier(
                dt_utils.REGIONAL_BENCHMARK,
                num_levels=80,  # default benchmark size in ICON Fortran
                keep_skip_values=False,
                backend=backend,
            ).grid
        case _:
            return simple_grid.simple_grid(backend=backend, num_levels=num_levels)

def construct_dummy_decomposition_info(
    grid: icon.IconGrid,
    backend: gtx_backend.Backend | None = None,
) -> definitions.DecompositionInfo:
    """
    A public helper function to construct a dummy decomposition info object for test cases
    refactored from grid_utils.py
    It can be removed once the grid manager returns the decomposition info
    """

    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)

    def _add_dimension(dim: gtx.Dimension) -> None:
        indices = data_alloc.index_field(grid, dim, backend=backend)
        owner_mask = xp.ones((grid.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

    decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
    _add_dimension(dims.EdgeDim)
    _add_dimension(dims.VertexDim)
    _add_dimension(dims.CellDim)

    return decomposition_info

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
    name, num_levels = parse_grid_spec(spec)

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

@pytest.fixture
def vertical_grid_params(
    lowest_layer_thickness: float,
    model_top_height: float,
    stretch_factor: float,
    damping_height: float,
) -> Dict[str, float]:
    """Group vertical grid configuration parameters into a dictionary."""
    return {
        "lowest_layer_thickness": lowest_layer_thickness,
        "model_top_height": model_top_height,
        "stretch_factor": stretch_factor,
        "damping_height": damping_height,
    }


@pytest.fixture
def metrics_factory_params(
    rayleigh_coeff: float,
    exner_expol: float,
    vwind_offctr: float,
    rayleigh_type: float,
) -> Dict[str, float]:
    """Group rayleigh damping configuration parameters into a dictionary."""
    return {
        "rayleigh_coeff": rayleigh_coeff,
        "exner_expol": exner_expol,
        "vwind_offctr": vwind_offctr,
        "rayleigh_type": rayleigh_type,
    }
