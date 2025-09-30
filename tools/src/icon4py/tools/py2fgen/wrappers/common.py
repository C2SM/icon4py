# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import contextlib
import functools
import logging
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Final, TypeAlias

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
from gt4py import eve
from gt4py._core import definitions as gt4py_definitions

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import base, horizontal as h_grid, icon
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    xp: Final = np
else:
    try:
        import cupy as cp

        xp = cp
    except ImportError:
        cp = None
        xp = np


NDArray: TypeAlias = np.ndarray | xp.ndarray

# TODO(havogt): import needed to register MultNodeRun in get_processor_properties, does the pattern make sense?
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


_BACKEND_MAP: dict[BackendIntEnum, gtx_typing.Backend | None] = {
    BackendIntEnum._GTFN_CPU: model_backends.BACKENDS["gtfn_cpu"],
    BackendIntEnum._GTFN_GPU: model_backends.BACKENDS["gtfn_gpu"],
}
with contextlib.suppress(NotImplementedError):  # dace backends might not be available
    _BACKEND_MAP |= {
        BackendIntEnum._DACE_CPU: model_backends.BACKENDS.get("dace_cpu"),
        BackendIntEnum._DACE_GPU: model_backends.BACKENDS.get("dace_gpu"),
    }


def select_backend(selector: BackendIntEnum, on_gpu: bool) -> gtx_typing.Backend:
    default_cpu = BackendIntEnum._GTFN_CPU
    default_gpu = BackendIntEnum._GTFN_GPU
    if selector == BackendIntEnum.DEFAULT:
        selector = BackendIntEnum.DEFAULT_GPU if on_gpu else BackendIntEnum.DEFAULT_CPU
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

    backend = _BACKEND_MAP.get(selector)
    assert backend is not None

    return backend


def cached_dummy_field_factory(
    allocator: gtx_typing.Backend,
) -> Callable[[str, gtx.Domain, gt4py_definitions.DType], gtx.Field]:
    # curried to exclude non-hashable backend from cache
    @functools.lru_cache(maxsize=20)
    def impl(_name: str, domain: gtx.Domain, dtype: gt4py_definitions.DType) -> gtx.Field:
        # _name is used to differentiate between different dummy fields
        return gtx.zeros(domain, dtype=dtype, allocator=allocator)  # type:ignore[arg-type]  # TODO(): fix type hint

    return impl


def adjust_fortran_indices(inp: NDArray) -> NDArray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return inp - 1


def shrink_to_dimension(
    sizes: dict[gtx.Dimension, int], tables: dict[gtx.FieldOffset, NDArray]
) -> dict[gtx.FieldOffset, NDArray]:
    """Shrink the neighbor tables from nproma size to the actual size of the grid."""
    return {k: v[: sizes[k.target[0]]] for k, v in tables.items()}


def add_origin(xp: ModuleType, table: NDArray) -> NDArray:
    return xp.column_stack((xp.arange(table.shape[0], dtype=xp.int32), table))


def get_nproma(tables: Iterable[NDArray]) -> int:
    tables = list(tables)
    nproma = tables[0].shape[0]
    if not all(table.shape[0] == nproma for table in tables):
        raise ValueError("All connectivity tables must have the same number of rows (nproma).")
    return nproma


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
    mean_cell_area: gtx.float64,  # type:ignore[name-defined]  # TODO(): fix type hint
    backend: gtx_typing.Backend,
) -> icon.IconGrid:
    log.debug("Constructing ICON Grid in Python...")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", vertical_size)

    log.debug("Offsetting Fortran connectivitity arrays by 1")

    xp = data_alloc.import_array_ns(backend)
    # TODO (halungge): icon has 0 values in these arrays in some places possibly where they don't use them in case
    #  they are accessed the `adjust_fortran_indices` might lead to crashes
    start_indices = {
        dims.CellDim: adjust_fortran_indices(cell_starts),
        dims.EdgeDim: adjust_fortran_indices(edge_starts),
        dims.VertexDim: adjust_fortran_indices(vertex_starts),
    }
    end_indices = {dims.CellDim: cell_ends, dims.EdgeDim: edge_ends, dims.VertexDim: vertex_ends}

    c2e = adjust_fortran_indices(c2e)
    c2v = adjust_fortran_indices(c2v)
    v2c = adjust_fortran_indices(v2c)
    e2v = adjust_fortran_indices(e2v)[
        :, 0:2
    ]  # slicing required for e2v as input data is actually e2c2v
    c2e2c = adjust_fortran_indices(c2e2c)
    v2e = adjust_fortran_indices(v2e)
    e2c2v = adjust_fortran_indices(e2c2v)
    e2c = adjust_fortran_indices(e2c)
    e2c2e = adjust_fortran_indices(e2c2e)

    # stacked arrays
    c2e2c0 = add_origin(xp, c2e2c)
    e2c2e0 = add_origin(xp, e2c2e)

    config = base.GridConfig(
        horizontal_config=base.HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_size=vertical_size,
        limited_area=limited_area,
        keep_skip_values=False,
    )

    neighbor_tables = {
        dims.C2E: c2e,
        dims.C2V: c2v,
        dims.E2C: e2c,
        dims.E2C2E: e2c2e,
        dims.C2E2C: c2e2c,
        dims.C2E2CO: c2e2c0,
        dims.E2C2EO: e2c2e0,
        dims.V2E: v2e,
        dims.E2V: e2v,
        dims.E2C2V: e2c2v,
        dims.V2C: v2c,
    }

    neighbor_tables = shrink_to_dimension(
        sizes={dims.EdgeDim: num_edges, dims.VertexDim: num_vertices, dims.CellDim: num_cells},
        tables=neighbor_tables,
    )
    domain_bounds_constructor = functools.partial(
        h_grid.get_start_end_idx_from_icon_arrays,
        start_indices=start_indices,
        end_indices=end_indices,
    )
    start_index, end_index = icon.get_start_and_end_index(domain_bounds_constructor)

    return icon.icon_grid(
        id_=grid_id,
        allocator=backend,
        config=config,
        neighbor_tables=neighbor_tables,
        start_index=start_index,
        end_index=end_index,
        global_properties=icon.GlobalGridParams(mean_cell_area=mean_cell_area),
    )


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
    c_glb_index = adjust_fortran_indices(c_glb_index)
    e_glb_index = adjust_fortran_indices(e_glb_index)
    v_glb_index = adjust_fortran_indices(v_glb_index)

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
