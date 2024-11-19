# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging as log

import gt4py._core.definitions as gtcore_defs
import gt4py.next.backend as gtx_backend

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import geometry, geometry_attributes as geometry_attrs, icon
from icon4py.model.common.utils import gt4py_field_allocation as alloc

from .grid_tests import utils as gridtest_utils


def is_cupy_device(backend: gtx_backend.Backend) -> bool:
    cuda_device_types = (
        gtcore_defs.DeviceType.CUDA,
        gtcore_defs.DeviceType.CUDA_MANAGED,
        gtcore_defs.DeviceType.ROCM,
    )
    return backend.allocator.__gt_device_type__ in cuda_device_types


def array_ns(try_cupy: bool):
    if try_cupy:
        try:
            import cupy as cp

            return cp
        except ImportError:
            log.warn("No cupy installed falling back to numpy for array_ns")
    import numpy as np

    return np


def import_array_ns(backend: gtx_backend.Backend):
    is_cupy_device(backend)
    return array_ns(is_cupy_device(backend))


grid_geometries = {}


# TODO @halungge: copied from test_geometry.py: should be remove from there.
#                 also check the imports. Should it rather go to the test_utils package?
def get_grid_geometry(backend: gtx_backend.Backend, grid_file: str) -> geometry.GridGeometry:
    on_gpu = is_cupy_device(backend)
    xp = array_ns(on_gpu)

    def construct_decomposition_info(grid: icon.IconGrid) -> definitions.DecompositionInfo:
        edge_indices = alloc.allocate_indices(dims.EdgeDim, grid)
        owner_mask = xp.ones((grid.num_edges,), dtype=bool)
        decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
        decomposition_info.with_dimension(dims.EdgeDim, edge_indices.ndarray, owner_mask)
        return decomposition_info

    def construct_grid_geometry(grid_file: str):
        gm = gridtest_utils.run_grid_manager(grid_file, on_gpu=on_gpu)
        grid = gm.grid
        decomposition_info = construct_decomposition_info(grid)
        geometry_source = geometry.GridGeometry(
            grid, decomposition_info, backend, gm.coordinates, gm.geometry, geometry_attrs.attrs
        )
        return geometry_source

    if not grid_geometries.get(grid_file):
        grid_geometries[grid_file] = construct_grid_geometry(grid_file)
    return grid_geometries[grid_file]
