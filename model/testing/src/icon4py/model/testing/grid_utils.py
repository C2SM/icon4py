# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from typing import Optional

import gt4py.next.backend as gtx_backend

import icon4py.model.common.grid.grid_manager as gm
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    cases,
    data_handling,
)


GLOBAL_GRIDFILE__WIP = "icon_grid_0013_R02B04_R.nc"
MCH_CH_R04B09_LEVELS__WIP = 65

grid_geometries = {}


def download_grid(grid: cases.Grid) -> pathlib.Path:
    assert grid.file_name
    assert grid.uri

    grid_file_path = get_grid_file_path(grid.file_name)
    if not grid_file_path.exists():
        data_handling.download_and_extract(
            grid.uri,
            grid_file_path.parent,
            grid_file_path.parent,
        )

    return grid_file_path


def get_grid_manager_for_experiment(
    experiment: cases.Experiment, backend: Optional[gtx_backend.Backend] = None
) -> gm.GridManager:
    return get_grid_manager(experiment.grid, experiment.num_levels, backend)


def get_grid_manager(
    grid: cases.Grid, num_levels: int, backend: Optional[gtx_backend.Backend] = None
) -> gm.GridManager:
    grid_file_path = download_grid(grid)
    gm = _run_grid_manager_for_file(grid_file_path, num_levels, backend)
    return gm


def get_grid_file_path(grid_file_path: pathlib.Path | str) -> pathlib.Path:
    return cases.GRIDS_PATH / grid_file_path


def get_grid_geometry(
    experiment: cases.Experiment, backend: Optional[gtx_backend.Backend]
) -> geometry.GridGeometry:
    register_name = f"{experiment.name}_{data_alloc.backend_name(backend)}"
    if register_name not in grid_geometries:
        grid_geometries[register_name] = _construct_grid_geometry(experiment, backend)

    return grid_geometries[register_name]


def _run_grid_manager_for_file(
    file_path: pathlib.Path, num_levels: int, backend: Optional[gtx_backend.Backend]
) -> gm.GridManager:
    """
    Load a grid file.

    Args:
        file: full path to the grid file
        num_levels: number of vertical levels, needed for IconGrid construction but independent from grid file
        backend: the gt4py backend we are running on
    """
    transformation = gm.ToZeroBasedIndexTransformation()
    manager = gm.GridManager(
        transformation,
        file_path,
        v_grid.VerticalGridConfig(num_levels=num_levels),
    )
    manager(backend=backend)
    manager.close()

    return manager


def _construct_dummy_decomposition_info(
    grid: icon.IconGrid, backend: Optional[gtx_backend.Backend]
) -> definitions.DecompositionInfo:
    on_gpu = data_alloc.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)

    decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
    for dim in [dims.EdgeDim, dims.VertexDim, dims.CellDim]:
        indices = data_alloc.index_field(grid, dim, backend=backend)
        owner_mask = xp.ones((grid.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

    return decomposition_info


def _construct_grid_geometry(
    experiment: cases.Experiment, backend: Optional[gtx_backend.Backend]
) -> geometry.GridGeometry:
    gm = get_grid_manager_for_experiment(experiment, backend=backend)
    grid = gm.grid
    decomposition_info = _construct_dummy_decomposition_info(grid, backend)
    geometry_source = geometry.GridGeometry(
        grid, decomposition_info, backend, gm.coordinates, gm.geometry, geometry_attrs.attrs
    )
    return geometry_source
