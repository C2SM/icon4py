# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    grid_manager as gm,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import config, data_handling, definitions, locking


grid_geometries: dict[str, geometry.GridGeometry] = {}


def get_grid_manager_from_experiment(
    experiment: definitions.Experiment,
    keep_skip_values: bool,
    backend: gtx_typing.Backend | None = None,
) -> gm.GridManager:
    return get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=experiment.num_levels,
        keep_skip_values=keep_skip_values,
        allocator=backend,
    )


def get_grid_manager_from_identifier(
    grid: definitions.GridDescription,
    num_levels: int,
    keep_skip_values: bool,
    allocator: gtx_typing.FieldBufferAllocationUtil | None,
) -> gm.GridManager:
    grid_file = _download_grid_file(grid)
    return get_grid_manager(
        grid_file, num_levels=num_levels, keep_skip_values=keep_skip_values, allocator=allocator
    )


def get_grid_manager(
    grid_file: pathlib.Path,
    num_levels: int,
    keep_skip_values: bool,
    allocator: gtx_typing.FieldBufferAllocationUtil | None,
) -> gm.GridManager:
    """
    Construct a GridManager instance for an ICON grid file.

    Args:
        grid_file: full path to the file
        num_levels: number of vertical levels, needed for IconGrid construction but independent from grid file
        keep_skip_values: whether to keep skip values
        backend: the gt4py Backend we are running on
    """
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(),
        grid_file,
        v_grid.VerticalGridConfig(num_levels=num_levels),
    )
    manager(allocator=allocator, keep_skip_values=keep_skip_values)
    return manager


def resolve_full_grid_file_name(grid: definitions.GridDescription) -> pathlib.Path:
    return definitions.grids_path().joinpath(grid.name, grid.file_name)


def _download_grid_file(grid: definitions.GridDescription) -> pathlib.Path:
    full_name = resolve_full_grid_file_name(grid)
    grid_directory = full_name.parent
    grid_directory.mkdir(parents=True, exist_ok=True)
    if config.ENABLE_GRID_DOWNLOAD:
        with locking.lock(grid_directory):
            if not full_name.exists():
                data_handling.download_and_extract(
                    grid.uri,
                    grid_directory,
                )
    else:
        # If grid download is disabled, we check if the file exists
        # without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not full_name.exists():
            raise FileNotFoundError(
                f"Grid file {full_name} does not exist and grid download is disabled."
            )
    return full_name


def construct_decomposition_info(
    grid: icon.IconGrid,
    allocator: gtx_typing.FieldBufferAllocationUtil | None = None,
) -> decomposition_defs.DecompositionInfo:
    on_gpu = device_utils.is_cupy_device(allocator)
    xp = data_alloc.array_ns(on_gpu)

    def _add_dimension(dim: gtx.Dimension) -> None:
        indices = data_alloc.index_field(grid, dim, allocator=allocator)
        owner_mask = xp.ones((grid.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)

    decomposition_info = decomposition_defs.DecompositionInfo()
    _add_dimension(dims.EdgeDim)
    _add_dimension(dims.VertexDim)
    _add_dimension(dims.CellDim)

    return decomposition_info


def get_grid_geometry(
    backend: gtx_typing.Backend | None, experiment: definitions.Experiment
) -> geometry.GridGeometry:
    register_name = "_".join((experiment.name, data_alloc.backend_name(backend)))

    def _construct_grid_geometry() -> geometry.GridGeometry:
        gm = get_grid_manager_from_identifier(
            experiment.grid,
            keep_skip_values=True,
            num_levels=experiment.num_levels,
            allocator=backend,
        )
        grid = gm.grid
        decomposition_info = construct_decomposition_info(grid, backend)
        geometry_source = geometry.GridGeometry(
            grid,
            decomposition_info,
            backend,
            gm.coordinates,
            gm.geometry_fields,
            geometry_attrs.attrs,
        )
        return geometry_source

    if not grid_geometries.get(register_name):
        grid_geometries[register_name] = _construct_grid_geometry()
    return grid_geometries[register_name]
