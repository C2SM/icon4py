# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
import pytest

import icon4py.model.common.grid.grid_manager as gm
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.test_utils import data_handling, datatest_utils as dt_utils
from icon4py.model.common.utils import gt4py_field_allocation as alloc


REGIONAL_GRIDFILE = "grid.nc"

GLOBAL_GRIDFILE = "icon_grid_0013_R02B04_R.nc"

GLOBAL_NUM_LEVELS = 60

MCH_CH_R04B09_LEVELS = 65

grid_geometries = {}


def get_grid_manager_for_experiment(
    experiment: str, backend: gtx_backend.Backend = None
) -> gm.GridManager:
    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        return _download_and_load_gridfile(
            dt_utils.R02B04_GLOBAL,
            num_levels=GLOBAL_NUM_LEVELS,
            backend=backend,
        )
    elif experiment == dt_utils.REGIONAL_EXPERIMENT:
        return _download_and_load_gridfile(
            dt_utils.REGIONAL_EXPERIMENT,
            num_levels=MCH_CH_R04B09_LEVELS,
            backend=backend,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def get_grid_manager(
    grid_file: str, num_levels: int, backend: gtx_backend.Backend
) -> gm.GridManager:
    return _download_and_load_gridfile(grid_file, num_levels=num_levels, backend=backend)


def _file_name(grid_file: str):
    match grid_file:
        case dt_utils.REGIONAL_EXPERIMENT:
            return REGIONAL_GRIDFILE
        case dt_utils.R02B04_GLOBAL:
            return GLOBAL_GRIDFILE
        case _:
            raise NotImplementedError(f"Add grid path for experiment '{grid_file}'")


def resolve_full_grid_file_name(grid_file_str: str) -> pathlib.Path:
    return dt_utils.GRIDS_PATH.joinpath(grid_file_str, _file_name(grid_file_str))


def _download_grid_file(file_path: str) -> pathlib.Path:
    full_name = resolve_full_grid_file_name(file_path)
    if not full_name.exists():
        data_handling.download_and_extract(
            dt_utils.GRID_URIS[file_path],
            full_name.parent,
            full_name.parent,
        )
    return full_name


def _run_grid_manager_for_file(
    file: str, num_levels: int, backend: gtx_backend.Backend
) -> gm.GridManager:
    """
    Load a grid file.
    Args:
        file: full path to the file (file + path)
        num_levels: number of vertical levels, needed for IconGrid construction but independent from grid file
        backend: the gt4py Backend we are running on

    Returns:

    """
    limited_area = is_regional(str(file))
    transformation = gm.ToZeroBasedIndexTransformation()
    manager = gm.GridManager(
        transformation,
        file,
        v_grid.VerticalGridConfig(num_levels=num_levels),
    )
    manager(backend=backend, limited_area=limited_area)
    manager.close()
    return manager


def _download_and_load_gridfile(
    file_path: str, num_levels: int, backend: gtx_backend.Backend
) -> gm.GridManager:
    grid_file = _download_grid_file(file_path)
    gm = _run_grid_manager_for_file(str(grid_file), num_levels, backend)
    return gm


def is_regional(experiment_or_file: str):
    return (
        dt_utils.REGIONAL_EXPERIMENT in experiment_or_file
        or REGIONAL_GRIDFILE in experiment_or_file
    )


def get_num_levels(experiment: str):
    return MCH_CH_R04B09_LEVELS if experiment == dt_utils.REGIONAL_EXPERIMENT else GLOBAL_NUM_LEVELS


def get_grid_geometry(
    backend: gtx_backend.Backend, experiment: str, grid_file: str
) -> geometry.GridGeometry:
    on_gpu = alloc.is_cupy_device(backend)
    xp = alloc.array_ns(on_gpu)
    num_levels = get_num_levels(experiment)

    def construct_decomposition_info(grid: icon.IconGrid) -> definitions.DecompositionInfo:
        def _add_dimension(dim: gtx.Dimension):
            indices = alloc.allocate_indices(dim, grid)
            owner_mask = xp.ones((grid.size[dim],), dtype=bool)
            decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

        decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
        _add_dimension(dims.EdgeDim)
        _add_dimension(dims.VertexDim)
        _add_dimension(dims.CellDim)

        return decomposition_info

    def construct_grid_geometry(grid_file: str):
        gm = _run_grid_manager_for_file(grid_file, backend=backend, num_levels=num_levels)
        grid = gm.grid
        decomposition_info = construct_decomposition_info(grid)
        geometry_source = geometry.GridGeometry(
            grid, decomposition_info, backend, gm.coordinates, gm.geometry, geometry_attrs.attrs
        )
        return geometry_source

    if not grid_geometries.get(grid_file):
        grid_geometries[grid_file] = construct_grid_geometry(
            str(resolve_full_grid_file_name(grid_file))
        )
    return grid_geometries[grid_file]


@pytest.fixture
def grid(request):
    return request.param
