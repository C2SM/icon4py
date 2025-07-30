# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
from typing import Optional

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
from gt4py._core import locking

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    grid_manager as gm,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import config, data_handling, datatest_utils as dt_utils


REGIONAL_GRIDFILE = "grid.nc"

GLOBAL_GRIDFILE = "icon_grid_0013_R02B04_R.nc"

GLOBAL_NUM_LEVELS = 60

MCH_CH_R04B09_LEVELS = 65

grid_geometries: dict[str, geometry.GridGeometry] = {}


def get_grid_manager_for_experiment(
    experiment: str, keep_skip_values: bool, backend: Optional[gtx_backend.Backend] = None
) -> gm.GridManager:
    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        return _download_and_load_gridfile(
            dt_utils.R02B04_GLOBAL,
            num_levels=GLOBAL_NUM_LEVELS,
            keep_skip_values=keep_skip_values,
            backend=backend,
        )
    elif experiment == dt_utils.REGIONAL_EXPERIMENT:
        return _download_and_load_gridfile(
            dt_utils.REGIONAL_EXPERIMENT,
            num_levels=MCH_CH_R04B09_LEVELS,
            keep_skip_values=keep_skip_values,
            backend=backend,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def get_grid_manager(
    grid_file: str, num_levels: int, keep_skip_values: bool, backend: Optional[gtx_backend.Backend]
) -> gm.GridManager:
    return _download_and_load_gridfile(
        file_path=grid_file,
        num_levels=num_levels,
        keep_skip_values=keep_skip_values,
        backend=backend,
    )


def _file_name(grid_file: str) -> str:
    match grid_file:
        case dt_utils.REGIONAL_EXPERIMENT:
            return REGIONAL_GRIDFILE
        case dt_utils.R02B04_GLOBAL:
            return GLOBAL_GRIDFILE
        case dt_utils.R02B07_GLOBAL:
            return "icon_grid_0023_R02B07_G.nc"
        case dt_utils.ICON_CH2_SMALL:
            return "mch_opr_r4b7_DOM01.nc"
        case dt_utils.REGIONAL_BENCHMARK:
            return "domain1_DOM01.nc"
        case _:
            raise NotImplementedError(f"Add grid path for experiment '{grid_file}'")


def resolve_full_grid_file_name(grid_file_str: str) -> pathlib.Path:
    return dt_utils.GRIDS_PATH.joinpath(grid_file_str, _file_name(grid_file_str))


def _download_grid_file(file_path: str) -> pathlib.Path:
    full_name = resolve_full_grid_file_name(file_path)
    grid_directory = full_name.parent
    grid_directory.mkdir(parents=True, exist_ok=True)
    with locking.lock(grid_directory):
        if not full_name.exists():
            if config.ENABLE_GRID_DOWNLOAD:
                data_handling.download_and_extract(
                    dt_utils.GRID_URIS[file_path],
                    grid_directory,
                )
            else:
                raise FileNotFoundError(
                    f"Grid file {full_name} does not exist and grid download is disabled."
                )
    return full_name


def _download_and_load_gridfile(
    file_path: str, num_levels: int, keep_skip_values: bool, backend: Optional[gtx_backend.Backend]
) -> gm.GridManager:
    """
    Load a grid file.
    Args:
        file: full path to the file (file + path)
        num_levels: number of vertical levels, needed for IconGrid construction but independent from grid file
        backend: the gt4py Backend we are running on

    Returns:

    """
    grid_file = _download_grid_file(file_path)
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(),
        grid_file,
        v_grid.VerticalGridConfig(num_levels=num_levels),
    )
    manager(backend=backend, keep_skip_values=keep_skip_values)
    return manager


def get_num_levels(experiment: str) -> int:
    return MCH_CH_R04B09_LEVELS if experiment == dt_utils.REGIONAL_EXPERIMENT else GLOBAL_NUM_LEVELS


def get_grid_geometry(
    backend: Optional[gtx_backend.Backend], experiment: str, grid_file: str
) -> geometry.GridGeometry:
    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)
    num_levels = get_num_levels(experiment)
    register_name = "_".join((experiment, data_alloc.backend_name(backend)))

    def _construct_dummy_decomposition_info(grid: icon.IconGrid) -> definitions.DecompositionInfo:
        def _add_dimension(dim: gtx.Dimension) -> None:
            indices = data_alloc.index_field(grid, dim, backend=backend)
            owner_mask = xp.ones((grid.size[dim],), dtype=bool)
            decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

        decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
        _add_dimension(dims.EdgeDim)
        _add_dimension(dims.VertexDim)
        _add_dimension(dims.CellDim)

        return decomposition_info

    def _construct_grid_geometry() -> geometry.GridGeometry:
        gm = _download_and_load_gridfile(
            grid_file, keep_skip_values=True, num_levels=num_levels, backend=backend
        )
        grid = gm.grid
        decomposition_info = _construct_dummy_decomposition_info(grid)
        geometry_source = geometry.GridGeometry(
            grid, decomposition_info, backend, gm.coordinates, gm.geometry, geometry_attrs.attrs
        )
        return geometry_source

    if not grid_geometries.get(register_name):
        grid_geometries[register_name] = _construct_grid_geometry()
    return grid_geometries[register_name]
