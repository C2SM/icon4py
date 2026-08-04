# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Distribution of decomposed model state for output.

In a distributed (MPI) run every rank holds only its local part of each field: the
entries it owns plus halo entries -- read-only copies of entries owned by neighboring
ranks. Before such fields can be written they must be reduced to owned entries (every
halo entry is written by the rank that owns it, where it is not a halo) and placed
correctly relative to the undecomposed global grid.
This module provides the strategies for that step, sitting between the
``FieldGroupMonitor`` (scheduling) and the writers (file format):

- :class:`SingleNodeDistribution`: no decomposition, the state passes through unchanged.
- :class:`GatherDistribution`: owned entries of all ranks are collected on the root rank
  and reassembled into global fields (identical to a single-rank run); only the root
  rank writes, using a serial writer.
- :class:`RankBlockDistribution`: every rank keeps its owned entries and writes them
  itself into a rank-contiguous block of a shared store (see :class:`RankBlock`).

All strategies operate solely on the communicator in ``process_props``; a future
compute/output communicator split only changes which ``process_props`` is passed in.

``prepare`` is collective: at a capture step it must be called on every rank of the
communicator (it performs the MPI communication). Ranks with ``writes_output == False``
contribute data but receive ``None``.
"""

import dataclasses
import logging
from typing import Final, Protocol

import gt4py.next as gtx
import numpy as np
import xarray as xr

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base
from icon4py.model.common.io import ugrid
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)

#: CF dimension name ("cell", "edge", "vertex") for each horizontal dimension.
HORIZONTAL_DIM_NAMES: Final[dict[gtx.Dimension, str]] = {
    dim: ugrid.dimension_mapping(dim, is_on_half_levels=False) for dim in dims.horizontal_dims()
}


class OutputDistribution(Protocol):
    """Strategy assembling the writable part of a (possibly decomposed) model state."""

    @property
    def writes_output(self) -> bool:
        """Whether this rank writes to the output file/store."""
        ...

    @property
    def output_horizontal_size(self) -> base.HorizontalGridSize:
        """Horizontal dimension sizes of the output file/store."""
        ...

    @property
    def rank_blocks(self) -> dict[str, "RankBlock"] | None:
        """Rank-block layout for writers doing per-rank region writes, None otherwise."""
        ...

    def prepare(self, state: dict[str, xr.DataArray]) -> dict[str, xr.DataArray] | None:
        """Assemble the state to write on this rank (None if this rank does not write).

        Collective: at a capture step every rank of the communicator must call this.
        """
        ...


@dataclasses.dataclass(frozen=True)
class SingleNodeDistribution:
    """Trivial distribution of a single-rank run: the full state is written as is."""

    horizontal_size: base.HorizontalGridSize

    @property
    def writes_output(self) -> bool:
        return True

    @property
    def output_horizontal_size(self) -> base.HorizontalGridSize:
        return self.horizontal_size

    @property
    def rank_blocks(self) -> dict[str, "RankBlock"] | None:
        return None

    def prepare(self, state: dict[str, xr.DataArray]) -> dict[str, xr.DataArray] | None:
        return state


@dataclasses.dataclass(frozen=True)
class RankBlock:
    """Rank-contiguous block of the padded horizontal axis of a shared output store.

    The horizontal axis of the store consists of ``comm_size`` blocks of the uniform
    ``size``: rank ``r`` writes its ``count`` owned entries to
    ``[r * size, r * size + count)``. The store is chunked so that no chunk crosses a
    block boundary -- by default with exactly one chunk per block -- hence concurrent
    writes of different ranks never touch the same chunk. An axis therefore never
    holds fewer chunks (or shard files) than ranks; the default one-chunk-per-block
    layout is that minimum. Dedicated IO ranks (planned for asynchronous output) will
    lower this floor to the IO-rank count, making chunk and shard counts freely
    tunable.

    Entries past ``count`` within a block are padding: they always read as the fill
    value (the zarr writer never writes them; the netCDF writer writes them
    explicitly, since its parallel collectives require every rank to participate in
    every write). The amount of padding per axis, ``padded_size - global_size``, is
    driven by the decomposition imbalance and the block alignment; the ratio is
    logged at construction. Because the axis is rank-ordered and padded, the
    variables of a rank-block store carry no UGRID mesh association: they are marked
    ``icon4py_layout = "rank_block"`` instead (see ``writers.LAYOUT_ATTRIBUTE``), and
    a consumer must reorder them by ``global_index`` before the mesh of the UGRID
    topology file applies.

    Attributes:
        start: first position of this rank's block on the store axis (``rank * size``).
        count: number of entries this rank owns (and writes).
        size: uniform block size: the maximum ``count`` over the ranks, rounded up to
            the block alignment of :class:`RankBlockDistribution`.
        padded_size: total size of the store axis (``comm_size * size``).
        global_size: number of entries of the undecomposed global grid.
        global_index: position of each of the ``count`` owned entries in the
            undecomposed global grid (a read-only array; padding entries carry no
            global index).
    """

    start: int
    count: int
    size: int
    padded_size: int
    global_size: int
    global_index: np.ndarray


def check_chunks_align_with_blocks(
    rank_blocks: dict[str, RankBlock], chunk_size: int, label: str
) -> None:
    """Reject a store chunk/shard size whose boundaries would cross rank-block boundaries.

    A store axis is chunked uniformly from position 0, so concurrent writes of
    different ranks stay in disjoint chunks (shards) only if every block size is a
    multiple of the chunk (shard) size -- which ``RankBlockDistribution`` guarantees
    when given the size as ``block_alignment``.

    Raises:
        ValueError: if a rank-block size is not a multiple of ``chunk_size``.
    """
    for dim_name, block in rank_blocks.items():
        if block.size % chunk_size != 0:
            raise ValueError(
                f"Invalid horizontal {label} size {chunk_size}: the rank-block size "
                f"{block.size} of dimension '{dim_name}' is not a multiple of it, so "
                f"{label}s of concurrently writing ranks would overlap."
            )


def _host_owner_data(
    decomposition_info: decomposition.DecompositionInfo, dim: gtx.Dimension
) -> tuple[np.ndarray, np.ndarray]:
    """Owner mask and owned global indices of a dimension, as host (numpy) arrays.

    The decomposition info may hold device (cupy) buffers on GPU backends, but
    everything downstream of here -- halo stripping, the MPI collectives and the
    writers -- operates on host arrays. The host copies are made once here, at
    distribution construction, instead of at every capture step.
    """
    mask = data_alloc.as_numpy(decomposition_info.owner_mask(dim))
    owned_global_index = data_alloc.as_numpy(
        decomposition_info.global_index(dim, decomposition.DecompositionInfo.EntryType.OWNED)
    ).astype(np.int64)
    return mask, owned_global_index


def _allgather_entry_counts(
    process_props: decomposition.ProcessProperties, count: int
) -> np.ndarray:
    """Owned-entry counts of all ranks (one entry per rank, in rank order)."""
    if process_props.is_single_rank():
        return np.asarray([count], dtype=np.int64)
    return np.asarray(process_props.comm.allgather(count), dtype=np.int64)


def _gather_entries(
    process_props: decomposition.ProcessProperties,
    local_entries: np.ndarray,
    *,
    entry_counts: np.ndarray,
) -> np.ndarray | None:
    """Concatenate the ranks' entries (leading-axis) on the root rank, in rank order.

    Returns the concatenation on the root rank and None on all other ranks.
    """
    if process_props.is_single_rank():
        return local_entries
    send = np.ascontiguousarray(local_entries)
    entry_elements = int(np.prod(send.shape[1:], dtype=np.int64))
    if process_props.rank == 0:
        gathered = np.empty((int(entry_counts.sum()), *send.shape[1:]), dtype=send.dtype)
        process_props.comm.Gatherv(send, [gathered, entry_counts * entry_elements], root=0)
        return gathered
    process_props.comm.Gatherv(send, None, root=0)
    return None


def _check_partition(
    process_props: decomposition.ProcessProperties,
    dim_name: str,
    owned_global_index: np.ndarray,
    entry_counts: np.ndarray,
) -> np.ndarray | None:
    """Check that the ranks' owned global indices partition the global grid.

    Collective: the owned global indices of all ranks are gathered on the root rank
    and verified to be a permutation of ``0..N-1`` -- overlapping or gappy owner
    masks would otherwise reassemble a plausible-looking but wrong global field. The
    verdict is broadcast so all ranks raise together instead of hanging in the next
    collective.

    Returns the gathered indices on the root rank (None on all other ranks), for
    reuse as insertion indices.

    Raises:
        ValueError: if the owned global indices are not a permutation of ``0..N-1``.
    """
    global_size = int(entry_counts.sum())
    gathered_index = _gather_entries(process_props, owned_global_index, entry_counts=entry_counts)
    is_partition = gathered_index is None or np.array_equal(
        np.sort(gathered_index), np.arange(global_size, dtype=np.int64)
    )
    if not process_props.is_single_rank():
        is_partition = process_props.comm.bcast(is_partition, root=0)
    if not is_partition:
        raise ValueError(
            f"Owner masks of dimension '{dim_name}' do not partition the global grid: "
            f"the owned global indices of all ranks are not a permutation of "
            f"0..{global_size - 1}."
        )
    return gathered_index


def _owned_entries(
    dim_name: str, owner_masks: dict[str, np.ndarray], field: xr.DataArray
) -> np.ndarray:
    """Drop halo entries from the leading (horizontal) axis of a field."""
    if dim_name not in owner_masks:
        raise ValueError(
            f"Cannot distribute field with leading dimension '{dim_name}': "
            f"expected one of {sorted(owner_masks)}."
        )
    # as_numpy: fields may still hold device buffers here (the writers transfer to
    # host with the same helper)
    data = data_alloc.as_numpy(field.data)
    mask = owner_masks[dim_name]
    if data.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Field of leading dimension '{dim_name}' has {data.shape[0]} entries, "
            f"but the owner mask has {mask.shape[0]}."
        )
    return data[mask]


class GatherDistribution:
    """Gathers the owned entries of decomposed fields onto the root rank.

    Halo entries are dropped using the owner masks of the decomposition; the owned
    entries of all ranks are then collected (MPI Gatherv) and placed at their global
    indices, reassembling each field exactly as in a single-rank run. Fields must have
    a horizontal first dimension; further dimensions (e.g. the vertical) are carried
    along unchanged. Only the root rank returns a state from ``prepare`` and writes,
    with a plain serial writer.

    This is root-memory-bound: every captured field is reassembled at its full global
    size on the root rank, so the peak memory there scales with the undecomposed grid,
    not with a rank's local part. ``RankBlockDistribution`` avoids this (each rank only
    ever holds its own block) at the cost of a rank-ordered, padded store.

    On a single-rank communicator no communication happens, but owner mask and global
    index are still applied (owned entries only, global order).
    """

    def __init__(
        self,
        process_props: decomposition.ProcessProperties,
        decomposition_info: decomposition.DecompositionInfo,
    ) -> None:
        self._process_props = process_props
        self._owner_masks: dict[str, np.ndarray] = {}
        self._entry_counts: dict[str, np.ndarray] = {}
        self._insert_index: dict[str, np.ndarray] = {}
        self._global_size: dict[str, int] = {}

        for dim, dim_name in HORIZONTAL_DIM_NAMES.items():
            mask, owned_global_index = _host_owner_data(decomposition_info, dim)
            entry_counts = _allgather_entry_counts(process_props, owned_global_index.shape[0])
            insert_index = _check_partition(
                process_props, dim_name, owned_global_index, entry_counts
            )
            self._owner_masks[dim_name] = mask
            self._entry_counts[dim_name] = entry_counts
            self._global_size[dim_name] = int(entry_counts.sum())
            if insert_index is not None:
                self._insert_index[dim_name] = insert_index

    @property
    def writes_output(self) -> bool:
        return self._process_props.rank == 0

    @property
    def output_horizontal_size(self) -> base.HorizontalGridSize:
        return base.HorizontalGridSize(
            num_cells=self._global_size[HORIZONTAL_DIM_NAMES[dims.CellDim]],
            num_edges=self._global_size[HORIZONTAL_DIM_NAMES[dims.EdgeDim]],
            num_vertices=self._global_size[HORIZONTAL_DIM_NAMES[dims.VertexDim]],
        )

    @property
    def rank_blocks(self) -> dict[str, RankBlock] | None:
        return None

    def prepare(self, state: dict[str, xr.DataArray]) -> dict[str, xr.DataArray] | None:
        gathered: dict[str, xr.DataArray] = {}
        for name, field in state.items():
            dim_name = str(field.dims[0])
            owned = _owned_entries(dim_name, self._owner_masks, field)
            entries = _gather_entries(
                self._process_props, owned, entry_counts=self._entry_counts[dim_name]
            )
            if entries is not None:
                global_field = np.empty(
                    (self._global_size[dim_name], *owned.shape[1:]), dtype=owned.dtype
                )
                global_field[self._insert_index[dim_name]] = entries
                gathered[name] = xr.DataArray(
                    global_field, dims=field.dims, attrs=dict(field.attrs)
                )
        return gathered if self.writes_output else None


class RankBlockDistribution:
    """Every rank writes its owned entries into its own block of a shared store.

    Halo entries are dropped using the owner masks; each rank then writes its owned
    entries itself, into the rank-contiguous block described by :class:`RankBlock`.
    At capture steps there is no data communication: ``prepare`` only strips halos
    (the copy this makes also detaches the output from the live model state).
    Construction gathers the owned global indices once to validate that the owner
    masks partition the global grid (see ``_check_partition``). The store's
    horizontal axes are padded to a uniform block size per rank; consumers recover
    the global order from the store's global-index coordinates.

    ``block_alignment`` rounds the uniform block size up to a multiple of the given
    value, so a store chunking (or sharding) of that granularity never crosses block
    boundaries (see ``check_chunks_align_with_blocks``); the extra positions are
    ordinary padding.
    """

    def __init__(
        self,
        process_props: decomposition.ProcessProperties,
        decomposition_info: decomposition.DecompositionInfo,
        block_alignment: int = 1,
    ) -> None:
        if block_alignment < 1:
            raise ValueError(
                f"Invalid block alignment {block_alignment}: must be a positive integer."
            )
        self._process_props = process_props
        self._owner_masks: dict[str, np.ndarray] = {}
        self._rank_blocks: dict[str, RankBlock] = {}

        for dim, dim_name in HORIZONTAL_DIM_NAMES.items():
            mask, owned_global_index = _host_owner_data(decomposition_info, dim)
            # ``RankBlock`` is frozen; keep the array it hands out immutable too
            owned_global_index.setflags(write=False)
            count = owned_global_index.shape[0]
            entry_counts = _allgather_entry_counts(process_props, count)
            _check_partition(process_props, dim_name, owned_global_index, entry_counts)
            max_count = int(entry_counts.max())
            size = (max_count + block_alignment - 1) // block_alignment * block_alignment
            global_size = int(entry_counts.sum())
            padded_size = size * process_props.comm_size
            if padded_size > global_size:
                log.info(
                    f"Rank-block axis '{dim_name}': {padded_size - global_size} of "
                    f"{padded_size} positions are padding "
                    f"({(padded_size - global_size) / padded_size:.1%}; the amount is "
                    f"driven by the decomposition imbalance and the block alignment)."
                )
            self._owner_masks[dim_name] = mask
            self._rank_blocks[dim_name] = RankBlock(
                start=process_props.rank * size,
                count=count,
                size=size,
                padded_size=padded_size,
                global_size=global_size,
                global_index=owned_global_index,
            )

    @property
    def writes_output(self) -> bool:
        return True

    @property
    def output_horizontal_size(self) -> base.HorizontalGridSize:
        """Padded sizes: the store's horizontal axes, not the global grid sizes."""
        return base.HorizontalGridSize(
            num_cells=self._rank_blocks[HORIZONTAL_DIM_NAMES[dims.CellDim]].padded_size,
            num_edges=self._rank_blocks[HORIZONTAL_DIM_NAMES[dims.EdgeDim]].padded_size,
            num_vertices=self._rank_blocks[HORIZONTAL_DIM_NAMES[dims.VertexDim]].padded_size,
        )

    @property
    def rank_blocks(self) -> dict[str, RankBlock]:
        return self._rank_blocks

    def prepare(self, state: dict[str, xr.DataArray]) -> dict[str, xr.DataArray] | None:
        return {
            name: xr.DataArray(
                _owned_entries(str(field.dims[0]), self._owner_masks, field),
                dims=field.dims,
                attrs=dict(field.attrs),
            )
            for name, field in state.items()
        }
