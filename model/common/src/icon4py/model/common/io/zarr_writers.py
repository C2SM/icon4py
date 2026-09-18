# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import functools
import logging
import pathlib
import types
import typing
import warnings
from typing import Self, TypeAlias

import numpy as np
import xarray as xr
import zarr

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import cf_utils, distributed, writers
from icon4py.model.common.states import metadata
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


#: One pending array write of a time slice: (array, index, data).
_VariableWrite: TypeAlias = tuple[zarr.Array, tuple[int | slice, ...], np.ndarray]  # noqa: UP040 [non-pep695-type-alias]


def _write_variables(writes: list[_VariableWrite]) -> None:
    """Perform the data writes of one appended time slice.

    The task an asynchronous append queues: pure chunk writes on arrays resolved by
    the submitter, so it touches no store metadata and performs no communication.
    """
    for variable, index, data in writes:
        variable[index] = data


class ZarrWriter:
    """
    Writer for zarr stores, mirroring the layout of the netCDF files.

    The store carries the same dimensions (time, level, half_level, cell, edge, vertex),
    coordinate variables and CF attributes as the netCDF files; arrays carry their
    dimension names natively (zarr format 3), so ``xarray.open_zarr`` reads the store
    like the corresponding netCDF file.

    Serial mode (``rank_blocks is None``): whole time slices are appended, exactly like
    the NETCDFWriter -- used for single-rank runs and on the root rank of gathered output.

    Rank-block mode (``rank_blocks`` given): every rank writes only its rank-contiguous
    block of the horizontal axes (see ``distributed.RankBlock``). The horizontal axes
    are chunked so that no chunk (or shard) crosses a rank-block boundary, so
    concurrent writes of different ranks never touch the same chunk file. The root rank
    performs all store-metadata operations (array creation, time-axis resizing); one
    broadcast per append orders them before the data writes of all ranks and carries
    a root-side failure to every rank (``_root_step``), so no rank hangs.
    ``global_index_<dim>`` coordinates map store positions to the undecomposed global
    grid; padding positions carry the fill value -1 (data padding reads as NaN for
    floating dtypes and as zero otherwise).

    ``horizontal_chunk_size`` overrides the horizontal chunk size (default: the whole
    axis in serial mode, one chunk per rank block in rank-block mode);
    ``horizontal_shard_size`` groups whole chunks into one storage file each (see
    ``FieldGroupIOConfig``). In rank-block mode the block size must be a multiple of
    the chunk/shard size (see ``distributed.check_chunks_align_with_blocks``).

    Asynchronous mode (``async_queue`` given): ``append`` performs the store-metadata
    operations and their broadcast (``_root_step``) synchronously as before, then
    queues the data writes instead of performing them, overlapping the writing with
    the caller's following work. Only local chunk writes run on the background thread -- all communication
    stays on the calling thread -- and the target arrays are resolved before queueing,
    so the thread reads no store metadata either (a concurrent time-axis resize of a
    later append only grows an array's shape and never touches the chunks of earlier
    slices). The caller must not mutate the appended data afterwards: the data of an
    asynchronous group must be decoupled from the model state (see
    ``FieldGroupMonitor.store``). ``close`` drains the queue, so the store is complete
    once it returns. A failed write surfaces on the failing rank at the next append or
    close; in rank-block mode the surviving ranks then block in the next
    ``_root_step`` broadcast -- the same failure envelope as a synchronous rank-local
    write error.
    """

    def __init__(
        self,
        *,
        file_name: pathlib.Path,
        vertical: v_grid.VerticalGrid,
        horizontal: base.HorizontalGridSize,
        time_properties: writers.TimeProperties,
        global_attrs: writers.GlobalFileAttributes,
        rank_blocks: dict[str, distributed.RankBlock] | None,
        process_props: decomposition.ProcessProperties,
        async_queue: writers.AsyncWriteQueue | None,
        horizontal_chunk_size: int | None = None,
        horizontal_shard_size: int | None = None,
    ):
        self._file_name = str(file_name)
        self._time_properties = time_properties
        self._vertical_params = vertical
        self._horizontal_sizes = writers.horizontal_axis_sizes(horizontal)
        self.attrs = global_attrs
        self._rank_blocks = rank_blocks
        self._horizontal_chunk_size = horizontal_chunk_size
        self._horizontal_shard_size = horizontal_shard_size
        # construction runs on every rank; an invalid layout must raise here, not on
        # the root rank alone inside a pre-barrier store operation
        if horizontal_shard_size is not None and (
            horizontal_chunk_size is None or horizontal_shard_size % horizontal_chunk_size != 0
        ):
            raise ValueError(
                f"Invalid horizontal shard size {horizontal_shard_size}: requires a "
                f"horizontal chunk size that divides it, got {horizontal_chunk_size}."
            )
        if rank_blocks is not None:
            alignment = horizontal_shard_size or horizontal_chunk_size
            if alignment is not None:
                label = "shard" if horizontal_shard_size is not None else "chunk"
                distributed.check_chunks_align_with_blocks(rank_blocks, alignment, label)
        self._process_props = process_props
        self._async_queue = async_queue
        self._group: zarr.Group | None = None
        # The append count doubles as the time index of the next slice. It is kept
        # locally (identical on all ranks: in rank-block mode append is called once per
        # capture step on every rank) instead of being derived from store metadata,
        # which a lagging rank could re-read only after the root already resized for a
        # later step.
        self._append_count = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    @functools.cached_property
    def num_levels(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0] - 1

    def _is_collective(self) -> bool:
        """Whether store operations are shared between the ranks (rank-block mode).

        In serial mode the writer never synchronizes, whatever communicator it holds:
        it is used by a single process (a single-rank run, or the root rank of a
        gathered write while the other ranks never touch the writer).
        """
        return self._rank_blocks is not None

    def _is_root(self) -> bool:
        return not self._is_collective() or self._process_props.rank == 0

    def _root_step(self, step: typing.Callable[[], None], what: str) -> None:
        """Run a store-metadata step on the root rank, then release the other ranks.

        Root-only store operations (creating the store, resizing arrays) may fail on
        the root alone -- a full disk, missing permissions, an existing store. The
        outcome is broadcast so every rank raises together instead of the non-root
        ranks waiting forever in the barrier that orders those operations before
        their data writes (compare ``IOMonitor._write_ugrid``). In serial mode the
        step simply runs (and raises) on the calling process.
        """
        failure: Exception | None = None
        if self._is_root():
            try:
                step()
            except Exception as err:
                failure = err
        if not self._is_collective() or self._process_props.is_single_rank():
            if failure is not None:
                raise failure
            return
        # the broadcast doubles as the barrier: no rank passes before the root is done
        message = self._process_props.comm.bcast(
            None if failure is None else f"{type(failure).__name__}: {failure}", root=0
        )
        if message is not None:
            raise RuntimeError(
                f"{what} '{self._file_name}' failed on the root rank: {message}"
            ) from failure

    def _horizontal_write_range(self, dim_name: str) -> slice:
        if self._rank_blocks is None:
            return slice(0, self._horizontal_sizes[dim_name])
        block = self._rank_blocks[dim_name]
        return slice(block.start, block.start + block.count)

    def _horizontal_chunk(self, dim_name: str) -> int:
        if self._horizontal_chunk_size is not None:
            return self._horizontal_chunk_size
        if self._rank_blocks is None:
            return self._horizontal_sizes[dim_name]
        return self._rank_blocks[dim_name].size

    def _horizontal_shards(self, middle_shape: tuple[int, ...]) -> tuple[int, ...] | None:
        """Shard shape of an array whose last axis is horizontal; None disables sharding."""
        if self._horizontal_shard_size is None:
            return None
        return (*middle_shape, self._horizontal_shard_size)

    def _data_array(self, name: str) -> zarr.Array:
        """Typed access to a store array (``Group.__getitem__`` may also yield groups)."""
        assert self._group is not None
        array = self._group[name]
        assert isinstance(array, zarr.Array)
        return array

    def initialize_dataset(self) -> None:
        self._root_step(self._create_store, "Creating the store")
        if self._is_collective():
            if not self._is_root():
                self._group = zarr.open_group(self._file_name, mode="r+")
            assert self._rank_blocks is not None
            for dim_name, block in self._rank_blocks.items():
                index_range = self._horizontal_write_range(dim_name)
                self._data_array(f"{writers.GLOBAL_INDEX_PREFIX}_{dim_name}")[index_range] = (
                    block.global_index
                )

    def _create_store(self) -> None:
        """Create the store with its coordinate arrays and attributes (root rank only)."""
        # mode "w-" refuses to overwrite an existing store; format 3 carries the
        # dimension names of every array natively (``dimension_names``), which
        # xarray reads like netCDF dimensions.
        group = zarr.open_group(self._file_name, mode="w-", zarr_format=3)
        log.info(f"Creating store {self._file_name}")
        group.attrs.update({k: str(v) for (k, v) in self.attrs.items()})

        # format 3 requires a fill value on every array (defaulting to the dtype's
        # zero); unlike a format-2 fill value it is not decoded as a missing-value
        # marker by xarray, so real coordinate values (e.g. level index 0) are safe
        times = group.create_array(
            writers.TIME, shape=(0,), chunks=(1,), dtype="f8", dimension_names=[writers.TIME]
        )
        times.attrs.update(writers.time_attributes(self._time_properties))

        levels = group.create_array(
            writers.MODEL_LEVEL,
            shape=(self.num_levels,),
            dtype=np.int32,
            dimension_names=[writers.MODEL_LEVEL],
        )
        levels[:] = np.arange(self.num_levels, dtype=np.int32)
        levels.attrs.update(metadata.LEVEL_ATTRIBUTES)

        half_levels = group.create_array(
            writers.MODEL_HALF_LEVEL,
            shape=(self.num_levels + 1,),
            dtype=np.int32,
            dimension_names=[writers.MODEL_HALF_LEVEL],
        )
        half_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)
        half_levels.attrs.update(metadata.HALF_LEVEL_ATTRIBUTES)

        heights = group.create_array(
            "height",
            shape=(self.num_levels + 1,),
            dtype=np.float64,
            dimension_names=[writers.MODEL_HALF_LEVEL],
        )
        heights[:] = data_alloc.as_numpy(self._vertical_params.interface_physical_height)
        heights.attrs.update(metadata.HEIGHT_ATTRIBUTES)

        if self._rank_blocks is not None:
            for dim_name, block in self._rank_blocks.items():
                global_index = group.create_array(
                    f"{writers.GLOBAL_INDEX_PREFIX}_{dim_name}",
                    shape=(block.padded_size,),
                    chunks=(self._horizontal_chunk(dim_name),),
                    shards=self._horizontal_shards(()),
                    dtype=np.int64,
                    fill_value=-1,
                    dimension_names=[dim_name],
                )
                global_index.attrs.update(
                    units="1",
                    long_name=(
                        f"position of each {dim_name} entry in the undecomposed global "
                        f"grid of {block.global_size} entries (-1 marks padding)"
                    ),
                )
        self._group = group

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None:
        """
        Append the fields to the store.

        Appends a time slice of the fields in the state_to_append dictionary to the store
        for the `model_time`, expanding the time coordinate by the `model_time`. In
        rank-block mode only this rank's horizontal block is written. In asynchronous
        mode the data writes are queued instead of performed (see the class docstring)
        and the data must not be mutated after this call.

        Args:
            state_to_append: fields to append
            model_time: time of the model state
        """
        assert self._group is not None
        canonical_slices, host_data = writers.canonicalize_time_slice(
            state_to_append, self._horizontal_sizes
        )
        time_pos = self._append_count

        def _extend_time_axis() -> None:
            assert self._group is not None
            times = self._data_array(writers.TIME)
            times.resize((time_pos + 1,))
            times[time_pos] = cf_utils.date2num(
                model_time,
                units=self._time_properties.units,
                calendar=self._time_properties.calendar,
            )
            for var_name, canonical_slice in canonical_slices.items():
                if var_name not in self._group:
                    self._create_variable(var_name, canonical_slice)
                variable = self._data_array(var_name)
                variable.resize((time_pos + 1, *variable.shape[1:]))

        self._root_step(_extend_time_axis, "Extending the time axis of")
        writes: list[_VariableWrite] = []
        for var_name, canonical_slice in canonical_slices.items():
            variable = self._data_array(var_name)
            horizontal_range = self._horizontal_write_range(str(canonical_slice.dims[-1]))
            middle = (slice(None),) * (len(canonical_slice.dims) - 1)
            index: tuple[int | slice, ...] = (time_pos, *middle, horizontal_range)
            writes.append((variable, index, host_data[var_name]))
        if self._async_queue is None:
            _write_variables(writes)
        else:
            self._async_queue.submit(functools.partial(_write_variables, writes))
        self._append_count += 1

    def _create_variable(self, var_name: str, canonical_slice: xr.DataArray) -> None:
        """Create the (empty along time) array of a data variable on the root rank."""
        assert self._group is not None
        horizontal_name = str(canonical_slice.dims[-1])
        shape = (0, *canonical_slice.shape[:-1], self._horizontal_sizes[horizontal_name])
        chunks = (1, *canonical_slice.shape[:-1], self._horizontal_chunk(horizontal_name))
        # NaN marks never-written positions (rank-block padding) but exists only for
        # floating dtypes; other dtypes fall back to the dtype's default fill value
        # (padding then reads as zero)
        fill_value = float("nan") if np.issubdtype(canonical_slice.dtype, np.floating) else None
        variable = self._group.create_array(
            var_name,
            shape=shape,
            chunks=chunks,
            shards=self._horizontal_shards(chunks[:-1]),
            dtype=canonical_slice.dtype,
            fill_value=fill_value,
            dimension_names=[writers.TIME, *(str(d) for d in canonical_slice.dims)],
        )
        variable.attrs.update(
            writers.data_variable_attributes(
                canonical_slice, rank_block_layout=self._rank_blocks is not None
            )
        )

    def close(self) -> None:
        if self._group is None:
            return
        if self._async_queue is not None:
            # the queued writes of this store must be on disk before its metadata is
            # consolidated
            self._async_queue.drain()
        if self._is_root():
            with warnings.catch_warnings():
                # consolidated metadata speeds up opening the store (single metadata
                # read for all arrays) but is a zarr-python extension of the format 3
                # specification; silence the spec-stability warning it emits on every
                # close (other readers simply ignore the extra metadata)
                warnings.filterwarnings(
                    "ignore", message="Consolidated metadata", category=UserWarning
                )
                zarr.consolidate_metadata(self._group.store)
        self._group = None
