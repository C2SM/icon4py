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
entries it owns plus halo entries duplicated from neighboring ranks. Before such fields
can be written they must be reduced to owned entries (halos are owned -- and written --
by their home rank) and placed correctly relative to the undecomposed global grid.
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
    def file_horizontal_size(self) -> base.HorizontalGridSize:
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
    def file_horizontal_size(self) -> base.HorizontalGridSize:
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
    size ``chunk`` (the maximum owned count over all ranks): rank ``r`` writes its
    ``count`` owned entries to ``[r * chunk, r * chunk + count)``. Since the store is
    chunked with exactly one chunk per block, concurrent writes of different ranks
    never touch the same chunk. Entries past ``count`` within a block are padding:
    they are never written and hold the fill value. ``global_index`` maps the block's
    entries to their positions in the undecomposed global grid of ``global_size``
    entries (padding entries carry no global index).
    """

    start: int
    count: int
    chunk: int
    padded_size: int
    global_size: int
    global_index: np.ndarray


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
        self._row_counts: dict[str, np.ndarray] = {}
        self._insert_index: dict[str, np.ndarray] = {}
        self._global_size: dict[str, int] = {}

        for dim, dim_name in HORIZONTAL_DIM_NAMES.items():
            mask = data_alloc.as_numpy(decomposition_info.owner_mask(dim))
            owned_global_index = data_alloc.as_numpy(
                decomposition_info.global_index(
                    dim, decomposition.DecompositionInfo.EntryType.OWNED
                )
            ).astype(np.int64)
            row_counts = self._allgather_counts(owned_global_index.shape[0])
            global_size = int(row_counts.sum())
            insert_index = self._gather_rows(owned_global_index, row_counts=row_counts)
            # the root rank checks the partition property; the verdict is broadcast so
            # all ranks raise together (a root-only raise would leave the other ranks
            # blocked in the next collective)
            is_partition = insert_index is None or np.array_equal(
                np.sort(insert_index), np.arange(global_size, dtype=np.int64)
            )
            if not self._process_props.is_single_rank():
                is_partition = self._process_props.comm.bcast(is_partition, root=0)
            if not is_partition:
                raise ValueError(
                    f"Owner masks of dimension '{dim_name}' do not partition the global grid: "
                    f"the owned global indices of all ranks are not a permutation of "
                    f"0..{global_size - 1}."
                )
            self._owner_masks[dim_name] = mask
            self._row_counts[dim_name] = row_counts
            self._global_size[dim_name] = global_size
            if insert_index is not None:
                self._insert_index[dim_name] = insert_index

    @property
    def writes_output(self) -> bool:
        return self._process_props.rank == 0

    @property
    def file_horizontal_size(self) -> base.HorizontalGridSize:
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
            rows = self._gather_rows(owned, row_counts=self._row_counts[dim_name])
            if rows is not None:
                global_field = np.empty(
                    (self._global_size[dim_name], *owned.shape[1:]), dtype=owned.dtype
                )
                global_field[self._insert_index[dim_name]] = rows
                gathered[name] = xr.DataArray(
                    global_field, dims=field.dims, attrs=dict(field.attrs)
                )
        return gathered if self.writes_output else None

    def _allgather_counts(self, count: int) -> np.ndarray:
        if self._process_props.is_single_rank():
            return np.asarray([count], dtype=np.int64)
        return np.asarray(self._process_props.comm.allgather(count), dtype=np.int64)

    def _gather_rows(self, local_rows: np.ndarray, *, row_counts: np.ndarray) -> np.ndarray | None:
        """Concatenate the ranks' rows (leading-axis entries) on the root rank, in rank order.

        Returns the concatenation on the root rank and None on all other ranks.
        """
        if self._process_props.is_single_rank():
            return local_rows
        send = np.ascontiguousarray(local_rows)
        row_elements = int(np.prod(send.shape[1:], dtype=np.int64))
        if self.writes_output:
            gathered = np.empty((int(row_counts.sum()), *send.shape[1:]), dtype=send.dtype)
            self._process_props.comm.Gatherv(send, [gathered, row_counts * row_elements], root=0)
            return gathered
        self._process_props.comm.Gatherv(send, None, root=0)
        return None


class RankBlockDistribution:
    """Every rank writes its owned entries into its own block of a shared store.

    Halo entries are dropped using the owner masks; each rank then writes its owned
    entries itself, into the rank-contiguous block described by :class:`RankBlock`.
    There is no data communication: ``prepare`` only strips halos (the copy this makes
    also detaches the output from the live model state). The store's horizontal axes
    are padded to a uniform block size per rank; consumers recover the global order
    from the store's global-index coordinates.
    """

    def __init__(
        self,
        process_props: decomposition.ProcessProperties,
        decomposition_info: decomposition.DecompositionInfo,
    ) -> None:
        self._process_props = process_props
        self._owner_masks: dict[str, np.ndarray] = {}
        self._rank_blocks: dict[str, RankBlock] = {}

        for dim, dim_name in HORIZONTAL_DIM_NAMES.items():
            mask = data_alloc.as_numpy(decomposition_info.owner_mask(dim))
            owned_global_index = data_alloc.as_numpy(
                decomposition_info.global_index(
                    dim, decomposition.DecompositionInfo.EntryType.OWNED
                )
            ).astype(np.int64)
            count = owned_global_index.shape[0]
            counts = (
                [count] if process_props.is_single_rank() else process_props.comm.allgather(count)
            )
            chunk = int(max(counts))
            self._owner_masks[dim_name] = mask
            self._rank_blocks[dim_name] = RankBlock(
                start=process_props.rank * chunk,
                count=count,
                chunk=chunk,
                padded_size=chunk * process_props.comm_size,
                global_size=int(sum(counts)),
                global_index=owned_global_index,
            )

    @property
    def writes_output(self) -> bool:
        return True

    @property
    def file_horizontal_size(self) -> base.HorizontalGridSize:
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
