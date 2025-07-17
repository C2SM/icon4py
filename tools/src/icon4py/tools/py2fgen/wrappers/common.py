# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# type: ignore

import functools
import logging
from typing import Callable, TypeAlias, Union

import gt4py.next as gtx
import numpy as np
from gt4py import eve
from gt4py._core import definitions as gt4py_definitions
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import base, icon


try:
    import dace  # type: ignore[import-untyped]
    from gt4py.next.program_processors.runners.dace import (
        run_dace_cpu_cached,
        run_dace_gpu_cached,
    )
except ImportError:
    from types import ModuleType
    from typing import Optional

    dace: Optional[ModuleType] = None  # type: ignore[no-redef] # definition needed here

try:
    import cupy as cp

    xp = cp
except ImportError:
    cp = None
    xp = np


NDArray: TypeAlias = Union[np.ndarray, xp.ndarray]

# TODO(havogt) import needed to register MultNodeRun in get_processor_properties, does the pattern make sense?
assert hasattr(mpi_decomposition, "get_multinode_properties")

log = logging.getLogger(__name__)


class BackendIntEnum(eve.IntEnum):
    DEFAULT = 0
    DEFAULT_CPU = 1
    DEFAULT_GPU = 2
    _GTFN_CPU = 11
    _GTFN_GPU = 12
    _DACE_CPU = 21
    _DACE_GPU = 22


_BACKEND_MAP = {
    BackendIntEnum._GTFN_CPU: run_gtfn_cached,
    BackendIntEnum._GTFN_GPU: run_gtfn_gpu_cached,
}
if dace:
    _BACKEND_MAP |= {
        BackendIntEnum._DACE_CPU: run_dace_cpu_cached,
        BackendIntEnum._DACE_GPU: run_dace_gpu_cached,
    }


def select_backend(selector: BackendIntEnum, on_gpu: bool) -> gtx_backend.Backend:
    default_cpu = BackendIntEnum._GTFN_CPU
    default_gpu = BackendIntEnum._GTFN_GPU
    if selector == BackendIntEnum.DEFAULT:
        if on_gpu:
            selector = BackendIntEnum.DEFAULT_GPU
        else:
            selector = BackendIntEnum.DEFAULT_CPU
    if selector == BackendIntEnum.DEFAULT_CPU:
        selector = default_cpu
    elif selector == BackendIntEnum.DEFAULT_GPU:
        selector = default_gpu

    if selector not in (
        BackendIntEnum._GTFN_CPU,
        BackendIntEnum._GTFN_GPU,
        BackendIntEnum._DACE_CPU,
        BackendIntEnum._DACE_GPU,
    ):
        raise ValueError(f"Invalid backend selector: {selector.name}")
    if on_gpu and selector in (BackendIntEnum._DACE_CPU, BackendIntEnum._GTFN_CPU):
        raise ValueError(f"Inconsistent backend selection: {selector.name} and on_gpu=True")
    if not on_gpu and selector in (BackendIntEnum._DACE_GPU, BackendIntEnum._GTFN_GPU):
        raise ValueError(f"Inconsistent backend selection: {selector.name} and on_gpu=False")

    assert selector in _BACKEND_MAP
    return _BACKEND_MAP[selector]


def cached_dummy_field_factory(
    allocator: gtx_backend.Backend,
) -> Callable[[str, gtx.Domain, gt4py_definitions.DType], gtx.Field]:
    # curried to exclude non-hashable backend from cache
    @functools.lru_cache(maxsize=20)
    def impl(_name: str, domain: gtx.Domain, dtype: gt4py_definitions.DType) -> gtx.Field:
        # _name is used to differentiate between different dummy fields
        return gtx.zeros(domain, dtype=dtype, allocator=allocator)

    return impl


def adjust_fortran_indices(inp: np.ndarray | NDArray, offset: int) -> np.ndarray | NDArray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return inp - offset


def construct_icon_grid(
    cell_starts: np.ndarray,
    cell_ends: np.ndarray,
    vertex_starts: np.ndarray,
    vertex_ends: np.ndarray,
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    c2e: NDArray,
    e2c: NDArray,
    c2e2c: NDArray,
    e2c2e: NDArray,
    e2v: NDArray,
    v2e: NDArray,
    v2c: NDArray,
    e2c2v: NDArray,
    c2v: NDArray,
    grid_id: str,
    num_vertices: int,
    num_cells: int,
    num_edges: int,
    vertical_size: int,
    limited_area: bool,
    on_gpu: bool,
):
    log.debug("Constructing ICON Grid in Python...")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", vertical_size)

    log.debug("Offsetting Fortran connectivitity arrays by 1")
    offset = 1

    xp = np if not on_gpu else cp

    cells_start_index = adjust_fortran_indices(cell_starts, offset)
    vertex_start_index = adjust_fortran_indices(vertex_starts, offset)
    edge_start_index = adjust_fortran_indices(edge_starts, offset)

    cells_end_index = cell_ends
    vertex_end_index = vertex_ends
    edge_end_index = edge_ends

    c2e = adjust_fortran_indices(c2e, offset)
    c2v = adjust_fortran_indices(c2v, offset)
    v2c = adjust_fortran_indices(v2c, offset)
    e2v = adjust_fortran_indices(e2v, offset)[
        :, 0:2
    ]  # slicing required for e2v as input data is actually e2c2v
    c2e2c = adjust_fortran_indices(c2e2c, offset)
    v2e = adjust_fortran_indices(v2e, offset)
    e2c2v = adjust_fortran_indices(e2c2v, offset)
    e2c = adjust_fortran_indices(e2c, offset)
    e2c2e = adjust_fortran_indices(e2c2e, offset)

    # stacked arrays
    c2e2c0 = xp.column_stack((xp.asarray(range(c2e2c.shape[0])), c2e2c))
    e2c2e0 = xp.column_stack((xp.asarray(range(e2c2e.shape[0])), e2c2e))

    config = base.GridConfig(
        horizontal_config=base.HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=on_gpu,
        keep_skip_values=False,
    )

    grid = (
        icon.IconGrid(id_=grid_id)
        .set_config(config)
        .set_start_end_indices(dims.VertexDim, vertex_start_index, vertex_end_index)
        .set_start_end_indices(dims.EdgeDim, edge_start_index, edge_end_index)
        .set_start_end_indices(dims.CellDim, cells_start_index, cells_end_index)
        .set_neighbor_tables(
            {
                dims.C2EDim: c2e,
                dims.C2VDim: c2v,
                dims.E2CDim: e2c,
                dims.E2C2EDim: e2c2e,
                dims.C2E2CDim: c2e2c,
                dims.C2E2CODim: c2e2c0,
                dims.E2C2EODim: e2c2e0,
            }
        )
        .set_neighbor_tables(
            {
                dims.V2EDim: v2e,
                dims.E2VDim: e2v,
                dims.E2C2VDim: e2c2v,
                dims.V2CDim: v2c,
            }
        )
    )

    grid.update_size_connectivities(
        {
            dims.ECVDim: grid.size[dims.EdgeDim] * grid.size[dims.E2C2VDim],
            dims.CEDim: grid.size[dims.CellDim] * grid.size[dims.C2EDim],
            dims.ECDim: grid.size[dims.EdgeDim] * grid.size[dims.E2CDim],
            dims.CECDim: grid.size[dims.CellDim] * grid.size[dims.C2E2CDim],
        }
    )

    return grid


def construct_decomposition(
    c_glb_index: np.ndarray,
    e_glb_index: np.ndarray,
    v_glb_index: np.ndarray,
    c_owner_mask: np.ndarray,
    e_owner_mask: np.ndarray,
    v_owner_mask: np.ndarray,
    num_cells: int,
    num_edges: int,
    num_vertices: int,
    num_levels: int,
    comm_id: int,
) -> tuple[
    definitions.ProcessProperties, definitions.DecompositionInfo, definitions.ExchangeRuntime
]:
    log.debug("Offsetting Fortran connectivitity arrays by 1")
    offset = 1

    c_glb_index = adjust_fortran_indices(c_glb_index, offset)
    e_glb_index = adjust_fortran_indices(e_glb_index, offset)
    v_glb_index = adjust_fortran_indices(v_glb_index, offset)

    c_owner_mask = c_owner_mask[:num_cells]
    e_owner_mask = e_owner_mask[:num_edges]
    v_owner_mask = v_owner_mask[:num_vertices]

    decomposition_info = (
        definitions.DecompositionInfo(
            klevels=num_levels, num_cells=num_cells, num_edges=num_edges, num_vertices=num_vertices
        )
        .with_dimension(dims.CellDim, c_glb_index, c_owner_mask)
        .with_dimension(dims.EdgeDim, e_glb_index, e_owner_mask)
        .with_dimension(dims.VertexDim, v_glb_index, v_owner_mask)
    )
    processor_props = definitions.get_processor_properties(definitions.MultiNodeRun(), comm_id)
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    return processor_props, decomposition_info, exchange
