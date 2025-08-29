# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import gt4py.next.backend as gtx_backend

from icon4py.model.common.decomposition import halo
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    grid_manager as gm,
    gridfile,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import (
    config,
    data_handling,
    datatest_utils as dt_utils,
    definitions,
    locking,
)


REGIONAL_GRIDFILE = "grid.nc"

GLOBAL_GRIDFILE = "icon_grid_0013_R02B04_R.nc"

GLOBAL_NUM_LEVELS = 60

MCH_CH_R04B09_LEVELS = 65

grid_geometries: dict[str, geometry.GridGeometry] = {}


def get_grid_manager_from_experiment(
    experiment: str, keep_skip_values: bool, backend: gtx_backend.Backend | None = None
) -> gm.GridManager:
    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        return get_grid_manager_from_identifier(
            dt_utils.R02B04_GLOBAL,
            num_levels=GLOBAL_NUM_LEVELS,
            keep_skip_values=keep_skip_values,
            backend=backend,
        )
    elif experiment == dt_utils.REGIONAL_EXPERIMENT:
        return get_grid_manager_from_identifier(
            dt_utils.REGIONAL_EXPERIMENT,
            num_levels=MCH_CH_R04B09_LEVELS,
            keep_skip_values=keep_skip_values,
            backend=backend,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def get_grid_manager_from_identifier(
    grid_file_identifier: str,
    num_levels: int,
    keep_skip_values: bool,
    backend: gtx_backend.Backend | None,
) -> gm.GridManager:
    grid_file = _download_grid_file(grid_file_identifier)
    return get_grid_manager(
        grid_file, num_levels=num_levels, keep_skip_values=keep_skip_values, backend=backend
    )


def get_grid_manager(
    filename: pathlib.Path,
    num_levels: int,
    keep_skip_values: bool,
    backend: gtx_backend.Backend | None,
) -> gm.GridManager:
    """
    Construct a GridManager instance for an ICON grid file.

    Args:
        filename: full path to the file
        num_levels: number of vertical levels, needed for IconGrid construction but independent from grid file
        keep_skip_values: whether to keep skip values
        backend: the gt4py Backend we are running on
    """
    manager = gm.GridManager(
        gridfile.ToZeroBasedIndexTransformation(),
        filename,
        v_grid.VerticalGridConfig(num_levels=num_levels),
    )
    manager(backend=backend, keep_skip_values=keep_skip_values)
    return manager


def _file_name(grid_file: str) -> str:  # noqa: PLR0911 [too-many-return-statements]
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
        case dt_utils.GAUSS3D_EXPERIMENT:
            return "Torus_Triangles_50000m_x_5000m_res500m.nc"
        case dt_utils.WEISMAN_KLEMP_EXPERIMENT:
            return "Torus_Triangles_50000m_x_5000m_res500m.nc"
        case _:
            raise NotImplementedError(f"Add grid path for experiment '{grid_file}'")


def resolve_full_grid_file_name(grid_file_identifier: str) -> pathlib.Path:
    return definitions.grids_path().joinpath(grid_file_identifier, _file_name(grid_file_identifier))


def _download_grid_file(grid_file_identifier: str) -> pathlib.Path:
    full_name = resolve_full_grid_file_name(grid_file_identifier)
    grid_directory = full_name.parent
    grid_directory.mkdir(parents=True, exist_ok=True)
    if config.ENABLE_GRID_DOWNLOAD:
        with locking.lock(grid_directory):
            if not full_name.exists():
                data_handling.download_and_extract(
                    dt_utils.GRID_URIS[grid_file_identifier],
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


def get_num_levels(experiment: str) -> int:
    return MCH_CH_R04B09_LEVELS if experiment == dt_utils.REGIONAL_EXPERIMENT else GLOBAL_NUM_LEVELS


def get_grid_geometry(
    backend: gtx_backend.Backend | None, experiment: str, grid_file: str
) -> geometry.GridGeometry:
    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)
    num_levels = get_num_levels(experiment)
    register_name = "_".join((experiment, data_alloc.backend_name(backend)))

    def _construct_grid_geometry() -> geometry.GridGeometry:
        gm = get_grid_manager_from_identifier(
            grid_file, keep_skip_values=True, num_levels=num_levels, backend=backend
        )
        grid = gm.grid
        dummy_halo_constructor = halo.NoHalos(
            horizontal_size=grid.config.horizontal_size, num_levels=num_levels, backend=backend
        )
        decomposition_info = dummy_halo_constructor(xp.zeros((grid.num_levels,), dtype=int))
        geometry_source = geometry.GridGeometry(
            grid, decomposition_info, backend, gm.coordinates, gm.geometry, geometry_attrs.attrs
        )
        return geometry_source

    if not grid_geometries.get(register_name):
        grid_geometries[register_name] = _construct_grid_geometry()
    return grid_geometries[register_name]
