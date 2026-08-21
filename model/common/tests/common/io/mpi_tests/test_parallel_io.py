# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Data-free parallel output tests on a synthetic decomposition.

A small global grid is partitioned across the ranks with unevenly sized parts, locally
shuffled global indices, fake halo entries and one completely empty rank (a situation
real experiment data does not provide, but large rank counts do). The real monitor ->
distribution -> writer stack then writes three capture steps with a file rollover in
between, and the root rank checks every output value exactly against the analytic
field ``1000 * global_index + level + 10 * step``.

Works with any rank count >= 2 (``mpirun -np 8 ... --with-mpi``), no test data needed.
"""

import datetime as dt
import itertools
import pathlib
import uuid

import gt4py.next as gtx
import numpy as np
import pytest
import xarray as xr
import zarr

import icon4py.model.common.exceptions as errors
from icon4py.model.common import dimension as dims, time as common_time
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.io import distributed, io as common_io, netcdf_writers, writers

from ...fixtures import process_props


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

NUM_LEVELS = 4
GLOBAL_SIZES = {dims.CellDim: 109, dims.EdgeDim: 75, dims.VertexDim: 41}
GRID_ID = uuid.UUID("00000000-0000-0000-0000-00000000c1c0")
NUM_STEPS = 3
#: variable name -> (CF dimension name, dimension) of the synthetic output fields
VARIABLES: dict[str, tuple[str, gtx.Dimension]] = {
    "air_density": ("cell", dims.CellDim),
    "normal_velocity": ("edge", dims.EdgeDim),
}


def synthetic_decomposition_info(
    process_props: decomp_defs.ProcessProperties,
) -> decomp_defs.DecompositionInfo:
    """Uneven partition with shuffled local order and fake halos; the last rank owns nothing."""
    empty_rank = process_props.comm_size - 1
    info = decomp_defs.DecompositionInfo()
    for dim, global_size in GLOBAL_SIZES.items():
        # deterministic, rank-independent seed (hash() is per-process randomized)
        rng = np.random.default_rng(seed=sum(ord(c) for c in dim.value))
        permutation = rng.permutation(global_size)
        working_ranks = [r for r in range(process_props.comm_size) if r != empty_rank]
        bounds = np.linspace(0, global_size, len(working_ranks) + 1).astype(int)
        owned_by_rank = {
            rank: permutation[start:stop]
            for rank, (start, stop) in zip(working_ranks, itertools.pairwise(bounds), strict=False)
        }
        owned_by_rank[empty_rank] = np.asarray([], dtype=np.int64)
        owned = owned_by_rank[process_props.rank]
        # fake halos: entries owned by the next working rank, as a real decomposition
        # would have (at 2 ranks there is only one working rank, which then donates its
        # own entries -- still halos in the sense that the owner mask is False on them)
        halo_donor = working_ranks[0] if process_props.rank == empty_rank else None
        if halo_donor is None:
            next_index = (working_ranks.index(process_props.rank) + 1) % len(working_ranks)
            halo_donor = working_ranks[next_index]
        halos = owned_by_rank[halo_donor][:2]
        local_global_index = np.concatenate([owned, halos]).astype(np.int64)
        owner_mask = np.zeros(local_global_index.shape[0], dtype=bool)
        owner_mask[: owned.shape[0]] = True
        local_shuffle = rng.permutation(local_global_index.shape[0])
        local_global_index = local_global_index[local_shuffle]
        owner_mask = owner_mask[local_shuffle]
        halo_levels = np.where(
            owner_mask,
            decomp_defs.DecompositionFlag.OWNED.value,
            decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL.value,
        )
        info.set_dimension(dim, local_global_index, owner_mask, halo_levels)
    return info


def expected_value(global_index: np.ndarray, level: int, step: int) -> np.ndarray:
    return 1000.0 * global_index + level + 10.0 * step


def synthetic_state(info: decomp_defs.DecompositionInfo, step: int) -> dict[str, xr.DataArray]:
    """Local state whose entries (halos included) carry their analytic global values."""
    state: dict[str, xr.DataArray] = {}
    for name, (dim_name, dim) in VARIABLES.items():
        local_global = info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.ALL)
        data = np.stack(
            [expected_value(local_global, level, step) for level in range(NUM_LEVELS)], axis=1
        )
        state[name] = xr.DataArray(
            data,
            dims=(dim_name, "level"),
            attrs=dict(
                units="1",
                standard_name=name,
                long_name=name,
                coordinates="lat lon",
                mesh="mesh",
                location=dim_name,
            ),
        )
    return state


def create_monitor(
    output_backend: common_io.OutputBackend,
    output_mode: common_io.OutputMode,
    distribution: distributed.OutputDistribution,
    process_props: decomp_defs.ProcessProperties,
    output_path: pathlib.Path,
    *,
    horizontal_chunk_size: int | None = None,
    horizontal_shard_size: int | None = None,
    asynchronous: bool | None = None,
) -> common_io.FieldGroupMonitor:
    config = common_io.FieldGroupIOConfig(
        basename="synthetic_output",
        variables=list(VARIABLES),
        output_interval=common_time.NumTimeSteps(1),
        timesteps_per_file=2,  # forces a file rollover at the third capture step
        backend=output_backend,
        mode=output_mode,
        horizontal_chunk_size=horizontal_chunk_size,
        horizontal_shard_size=horizontal_shard_size,
        asynchronous=asynchronous,
    )
    vertical = v_grid.VerticalGrid(
        v_grid.VerticalGridConfig(num_levels=NUM_LEVELS),
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, NUM_LEVELS + 1)),
        vct_b=None,
    )
    return common_io.FieldGroupMonitor(
        config=config,
        vertical=vertical,
        distribution=distribution,
        grid_id=GRID_ID,
        dtime=common_time.RelativeTime(hours=1),
        process_props=process_props,
        output_path=output_path,
    )


def assert_dataset_is_exact(
    dataset: xr.Dataset,
    num_slices: int,
    first_step: int,
    global_index: dict[str, np.ndarray],
) -> None:
    """Reassemble to global order and compare exactly (float64 arithmetic throughout)."""
    assert dataset.sizes["time"] == num_slices
    for name, (dim_name, dim) in VARIABLES.items():
        values = dataset[name].values
        index = global_index[dim_name]
        valid = index >= 0
        assert int(valid.sum()) == GLOBAL_SIZES[dim]
        for slice_pos in range(num_slices):
            step = first_step + slice_pos
            reassembled = np.empty((NUM_LEVELS, GLOBAL_SIZES[dim]))
            reassembled[:, index[valid]] = values[slice_pos][:, valid]
            expected = np.stack(
                [
                    expected_value(np.arange(GLOBAL_SIZES[dim]), level, step)
                    for level in range(NUM_LEVELS)
                ]
            )
            assert np.array_equal(reassembled, expected), f"'{name}' differs at step {step}"
        if valid.shape[0] > int(valid.sum()):
            assert np.all(np.isnan(values[:, :, ~valid])), f"'{name}' padding is not NaN"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "output_backend, output_mode, chunking, asynchronous",
    [
        (common_io.OutputBackend.NETCDF, common_io.OutputMode.GATHER, None, None),
        # the unconfigured zarr cases write asynchronously (the default); one explicit
        # synchronous case keeps the fallback path covered in distributed runs
        (common_io.OutputBackend.ZARR, common_io.OutputMode.GATHER, None, None),
        (common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED, None, None),
        (common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED, None, False),
        # sub-chunked, sharded rank blocks (block size rounded up to the shard size)
        (common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED, (8, 16), None),
        pytest.param(
            common_io.OutputBackend.NETCDF,
            common_io.OutputMode.DISTRIBUTED,
            None,
            None,
            marks=pytest.mark.skipif(
                netcdf_writers.missing_parallel_support() is not None,
                reason=(
                    "needs an MPI-parallel netCDF4 installation (PyPI wheels are serial "
                    "builds; see 'Parallel netCDF' in 'icon4py.model.common.io')"
                ),
            ),
        ),
    ],
)
def test_parallel_output_synthetic_decomposition(
    process_props: decomp_defs.ProcessProperties,
    *,
    output_backend: common_io.OutputBackend,
    output_mode: common_io.OutputMode,
    chunking: tuple[int, int] | None,
    asynchronous: bool | None,
    tmp_path: pathlib.Path,
) -> None:
    output_path = pathlib.Path(
        process_props.comm.bcast(str(tmp_path) if process_props.rank == 0 else None, root=0)
    )
    chunk_size, shard_size = chunking if chunking is not None else (None, None)
    info = synthetic_decomposition_info(process_props)
    distribution: distributed.OutputDistribution = (
        distributed.GatherDistribution(process_props, info)
        if output_mode == common_io.OutputMode.GATHER
        else distributed.RankBlockDistribution(
            process_props, info, block_alignment=shard_size or chunk_size or 1
        )
    )
    monitor = create_monitor(
        output_backend,
        output_mode,
        distribution,
        process_props,
        output_path,
        horizontal_chunk_size=chunk_size,
        horizontal_shard_size=shard_size,
        asynchronous=asynchronous,
    )

    for step in range(NUM_STEPS):
        model_time = dt.datetime(2000, 1, 1) + step * dt.timedelta(hours=1)
        monitor.store(synthetic_state(info, step), model_time)
    monitor.close()
    process_props.comm.Barrier()
    if process_props.rank != 0:
        return

    suffix = common_io.FILE_SUFFIXES[output_backend]
    files = sorted(output_path.glob(f"synthetic_output_*{suffix}"))
    assert len(files) == 2, f"expected two files after the rollover, got: {files}"
    identity = {name: np.arange(GLOBAL_SIZES[dim]) for name, dim in VARIABLES.values()}
    for path, num_slices, first_step in ((files[0], 2, 0), (files[1], 1, 2)):
        if output_backend == common_io.OutputBackend.NETCDF:
            dataset = xr.open_dataset(path, decode_times=False)
        else:
            dataset = xr.open_zarr(path, decode_times=False, mask_and_scale=False)
        with dataset:
            if output_mode == common_io.OutputMode.DISTRIBUTED:
                global_index = {
                    name: dataset[f"{writers.GLOBAL_INDEX_PREFIX}_{name}"].values
                    for name, _ in VARIABLES.values()
                }
            else:
                global_index = identity
            assert_dataset_is_exact(dataset, num_slices, first_step, global_index)
        if chunking is not None:
            # the configured layout must reach the store through the production path
            group = zarr.open_group(path, mode="r")
            air_density = group["air_density"]
            global_index_cell = group[f"{writers.GLOBAL_INDEX_PREFIX}_cell"]
            assert isinstance(air_density, zarr.Array)
            assert isinstance(global_index_cell, zarr.Array)
            assert air_density.chunks == (1, NUM_LEVELS, chunk_size)
            assert air_density.shards == (1, NUM_LEVELS, shard_size)
            assert global_index_cell.chunks == (chunk_size,)
            assert global_index_cell.shards == (shard_size,)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_gather_mode_overwrite_refusal_raises_on_all_ranks(
    process_props: decomp_defs.ProcessProperties,
    tmp_path: pathlib.Path,
) -> None:
    """The overwrite refusal must raise collectively, also in gather mode.

    Only the root rank writes in gather mode; a root-only raise would leave the other
    ranks blocked in the next collective, hanging the job instead of aborting it.
    """
    output_path = pathlib.Path(
        process_props.comm.bcast(str(tmp_path) if process_props.rank == 0 else None, root=0)
    )
    info = synthetic_decomposition_info(process_props)
    distribution = distributed.GatherDistribution(process_props, info)
    monitor = create_monitor(
        common_io.OutputBackend.NETCDF,
        common_io.OutputMode.GATHER,
        distribution,
        process_props,
        output_path,
    )
    if process_props.rank == 0:
        output_path.joinpath("synthetic_output_0001.nc").touch()
    process_props.comm.Barrier()
    with pytest.raises(errors.InvalidConfigError, match="already exists"):
        monitor.store(synthetic_state(info, 0), dt.datetime(2000, 1, 1))
