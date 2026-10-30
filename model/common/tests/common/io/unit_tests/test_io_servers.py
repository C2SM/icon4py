# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""MPI-free tests of the dedicated-IO-rank output path.

The compute<->IO inter-communicator is replaced by an in-process stand-in whose
mailboxes are queues, so several compute ranks and one IO rank run in one process
(the IO rank on a thread). This exercises the whole protocol -- setup, capture
headers and data, shutdown, mismatch detection -- and the IO side's monitor stack on
a served (permuted, halo-free) decomposition, including the single-IO-rank case where
the store must not be written as if it were the undecomposed grid.
"""

import collections
import copy
import dataclasses
import datetime as dt
import itertools
import pathlib
import queue
import threading
import uuid
from collections.abc import Sequence

import gt4py.next as gtx
import numpy as np
import pytest
import xarray as xr

import icon4py.model.common.exceptions as errors
from icon4py.model.common import dimension as dims, time as common_time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import io as common_io, io_servers, writers

from .. import utils as test_io_utils


NUM_LEVELS = 3
NUM_COMPUTE_RANKS = 3
GLOBAL_SIZES = {dims.CellDim: 17, dims.EdgeDim: 11, dims.VertexDim: 6}
GRID_ID = uuid.UUID("00000000-0000-0000-0000-00000000c1c1")
DTIME = common_time.RelativeTime(hours=1)
#: variable name -> (CF dimension name, number of levels) of the synthetic output
#: fields; 0 levels is a purely horizontal (surface) field
VARIABLES: dict[str, tuple[str, int]] = {
    "air_density": ("cell", NUM_LEVELS),
    "normal_velocity": ("edge", NUM_LEVELS),
    "surface_pressure": ("cell", 0),
}


# ------------------------------------------------------------------------------------
# In-process stand-in for the inter-communicator
# ------------------------------------------------------------------------------------


class _CompletedRequest:
    def Wait(self) -> None:
        pass


class _FakeIntercomm:
    """Inter-communicator stand-in: one queue per (receiver role, receiver, sender, tag).

    Sends copy their payload (as MPI does eventually) and complete immediately;
    receives block on the queue, so the IO rank can run on a thread.
    """

    def __init__(
        self,
        mailboxes: dict[tuple[str, int, int, int], queue.Queue],
        role: str,
        rank: int,
        local_size: int,
        remote_size: int,
    ):
        self._mailboxes = mailboxes
        self._role = role
        self._rank = rank
        self._local_size = local_size
        self._remote_size = remote_size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._local_size

    def Get_remote_size(self) -> int:
        return self._remote_size

    def _outbox(self, dest: int, tag: int) -> queue.Queue:
        remote_role = "io" if self._role == "compute" else "compute"
        return self._mailboxes[(remote_role, dest, self._rank, tag)]

    def _inbox(self, source: int, tag: int) -> queue.Queue:
        return self._mailboxes[(self._role, self._rank, source, tag)]

    def send(self, obj: object, dest: int, tag: int) -> None:
        self._outbox(dest, tag).put(copy.deepcopy(obj))

    def isend(self, obj: object, dest: int, tag: int) -> _CompletedRequest:
        self.send(obj, dest, tag)
        return _CompletedRequest()

    def recv(self, source: int, tag: int) -> object:
        return self._inbox(source, tag).get()

    def Isend(self, buffer: np.ndarray, dest: int, tag: int) -> _CompletedRequest:
        self._outbox(dest, tag).put(np.array(buffer, copy=True))
        return _CompletedRequest()

    def Recv(self, buffer: np.ndarray, source: int, tag: int) -> None:
        buffer[...] = self._inbox(source, tag).get()


def fake_intercomms(
    num_compute_ranks: int, num_io_ranks: int
) -> tuple[list[_FakeIntercomm], list[_FakeIntercomm]]:
    mailboxes: dict[tuple[str, int, int, int], queue.Queue] = collections.defaultdict(queue.Queue)
    compute = [
        _FakeIntercomm(mailboxes, "compute", rank, num_compute_ranks, num_io_ranks)
        for rank in range(num_compute_ranks)
    ]
    io = [
        _FakeIntercomm(mailboxes, "io", rank, num_io_ranks, num_compute_ranks)
        for rank in range(num_io_ranks)
    ]
    return compute, io


# ------------------------------------------------------------------------------------
# Synthetic decomposition and state
# ------------------------------------------------------------------------------------


def distributed_dim(dim_name: str) -> gtx.Dimension:
    return {"cell": dims.CellDim, "edge": dims.EdgeDim, "vertex": dims.VertexDim}[dim_name]


def synthetic_decomposition_info(compute_rank: int) -> decomp_defs.DecompositionInfo:
    """Uneven partition with shuffled local order and fake halos; the last rank owns nothing."""
    empty_rank = NUM_COMPUTE_RANKS - 1
    info = decomp_defs.DecompositionInfo()
    for dim, global_size in GLOBAL_SIZES.items():
        rng = np.random.default_rng(seed=sum(ord(c) for c in dim.value))
        permutation = rng.permutation(global_size)
        working_ranks = [r for r in range(NUM_COMPUTE_RANKS) if r != empty_rank]
        bounds = np.linspace(0, global_size, len(working_ranks) + 1).astype(int)
        owned_by_rank = {
            rank: permutation[start:stop]
            for rank, (start, stop) in zip(working_ranks, itertools.pairwise(bounds), strict=False)
        }
        owned_by_rank[empty_rank] = np.asarray([], dtype=np.int64)
        owned = owned_by_rank[compute_rank]
        halos = owned_by_rank[
            working_ranks[0] if compute_rank != working_ranks[0] else working_ranks[1]
        ][:2]
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
    for name, (dim_name, num_levels) in VARIABLES.items():
        local_global = info.global_index(
            distributed_dim(dim_name), decomp_defs.DecompositionInfo.EntryType.ALL
        )
        if num_levels:
            data = np.stack(
                [expected_value(local_global, level, step) for level in range(num_levels)],
                axis=1,
            )
        else:
            data = expected_value(local_global, 0, step)
        state[name] = xr.DataArray(
            data,
            dims=(dim_name, "level") if num_levels else (dim_name,),
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


def io_config(
    output_path: pathlib.Path,
    backend: common_io.OutputBackend,
    mode: common_io.OutputMode,
    output_interval: common_io.OutputInterval = common_time.NumTimeSteps(1),
) -> common_io.IOConfig:
    return common_io.IOConfig(
        output_path=output_path,
        field_groups=[
            common_io.FieldGroupIOConfig(
                basename="synthetic_output",
                variables=list(VARIABLES),
                output_interval=output_interval,
                timesteps_per_file=2,  # forces a file rollover at the third capture step
                backend=backend,
                mode=mode,
            )
        ],
        io_ranks_per_node=1,
    )


def horizontal_size() -> base.HorizontalGridSize:
    return base.HorizontalGridSize(
        num_cells=GLOBAL_SIZES[dims.CellDim],
        num_edges=GLOBAL_SIZES[dims.EdgeDim],
        num_vertices=GLOBAL_SIZES[dims.VertexDim],
    )


def vertical_grid() -> v_grid.VerticalGrid:
    return v_grid.VerticalGrid(
        v_grid.VerticalGridConfig(num_levels=NUM_LEVELS),
        vct_a=gtx.as_field((dims.KHalfDim,), np.linspace(12000.0, 0.0, NUM_LEVELS + 1)),
        vct_b=None,
    )


class ServerThread:
    """Runs one IO rank (setup, monitor construction, serve loop) on a thread.

    Exceptions of the IO rank are re-raised by ``join``.
    """

    def __init__(self, intercomm: _FakeIntercomm):
        self.server = io_servers.IOServer(
            intercomm,  # type: ignore[arg-type]  # in-process stand-in
            served_ranks=range(NUM_COMPUTE_RANKS),
        )
        self.setup: io_servers.ServerSetup | None = None
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.setup = self.server.receive_setup()
            io_monitor = common_io.IOMonitor(
                config=self.setup.config,
                vertical_size=vertical_grid(),
                horizontal_size=self.setup.horizontal_size,
                grid_file_name=test_io_utils.grid_file,
                grid_id=self.setup.grid_id,
                dtime=DTIME,
                process_props=decomp_defs.SingleNodeProcessProperties(),
                decomposition_info=self.setup.decomposition_info,
            )
            self.server.serve(io_monitor)
        except BaseException as err:
            self._failure = err

    def join(self) -> None:
        self._thread.join(timeout=60)
        assert not self._thread.is_alive(), "the IO rank did not finish"
        if self._failure is not None:
            raise self._failure


def make_clients(
    config: common_io.IOConfig, compute_intercomms: list[_FakeIntercomm]
) -> tuple[list[io_servers.IOClient], list[decomp_defs.DecompositionInfo]]:
    infos = [synthetic_decomposition_info(rank) for rank in range(NUM_COMPUTE_RANKS)]
    clients = [
        io_servers.IOClient(
            config=config,
            dtime=DTIME,
            grid_id=GRID_ID,
            horizontal_size=horizontal_size(),
            decomposition_info=info,
            intercomm=intercomm,  # type: ignore[arg-type]  # in-process stand-in
            io_rank=0,
        )
        for info, intercomm in zip(infos, compute_intercomms, strict=True)
    ]
    return clients, infos


def assert_dataset_is_exact(
    dataset: xr.Dataset, steps: Sequence[int], global_index: dict[str, np.ndarray]
) -> None:
    """Reassemble to global order and compare exactly against the analytic values."""
    assert dataset.sizes["time"] == len(steps)
    for name, (dim_name, num_levels) in VARIABLES.items():
        values = dataset[name].values
        index = global_index[dim_name]
        valid = index >= 0
        global_size = GLOBAL_SIZES[distributed_dim(dim_name)]
        assert int(valid.sum()) == global_size
        global_range = np.arange(global_size)
        for slice_pos, step in enumerate(steps):
            # a (levels, horizontal) slice, or a (horizontal,) one for a surface field
            reassembled = np.empty((*values.shape[1:-1], global_size))
            reassembled[..., index[valid]] = values[slice_pos][..., valid]
            expected = (
                np.stack([expected_value(global_range, level, step) for level in range(num_levels)])
                if num_levels
                else expected_value(global_range, 0, step)
            )
            assert np.array_equal(reassembled, expected), f"'{name}' differs at step {step}"


# ------------------------------------------------------------------------------------
# End-to-end through the stand-in
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend, mode",
    [
        (common_io.OutputBackend.NETCDF, common_io.OutputMode.GATHER),
        (common_io.OutputBackend.NETCDF, common_io.OutputMode.DISTRIBUTED),
        (common_io.OutputBackend.ZARR, common_io.OutputMode.GATHER),
        (common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED),
    ],
)
def test_io_rank_writes_the_shipped_captures_exactly(
    backend: common_io.OutputBackend, mode: common_io.OutputMode, tmp_path: pathlib.Path
) -> None:
    """Three compute ranks ship every second step to one IO rank, which writes them.

    A single IO rank writes on a single-rank communicator, but its state is a
    permutation of the global grid: it must go through the mode's distribution
    (reordering to global order for gathered output; a one-block rank-block store
    otherwise), never through the single-rank pass-through.
    """
    compute_intercomms, (io_intercomm,) = fake_intercomms(NUM_COMPUTE_RANKS, 1)
    config = io_config(tmp_path, backend, mode, output_interval=common_time.NumTimeSteps(2))
    server = ServerThread(io_intercomm)
    clients, infos = make_clients(config, compute_intercomms)
    assert all(client.io_rank == 0 for client in clients)

    num_steps = 6  # captures at steps 0, 2 and 4 -> a rollover after two
    for step in range(num_steps):
        model_time = dt.datetime(2000, 1, 1) + step * dt.timedelta(hours=1)
        for client, info in zip(clients, infos, strict=True):
            expected = set(VARIABLES) if step % 2 == 0 else set()
            assert client.variables_at_capture_time() == expected
            client.store(synthetic_state(info, step), model_time)
    for client in clients:
        client.close()
        client.close()  # idempotent
    server.join()

    assert server.setup is not None
    assert server.server.served_ranks == tuple(range(NUM_COMPUTE_RANKS))
    assert not server.setup.decomposition_info.is_identity()
    for client in clients:
        seconds = client.phase_seconds
        assert set(seconds) == {
            common_io.PHASE_DISTRIBUTE,
            common_io.PHASE_SEND,
            common_io.PHASE_SEND_WAIT,
        }
        assert all(len(values) == 3 for values in seconds.values())

    suffix = common_io.FILE_SUFFIXES[backend]
    files = sorted(tmp_path.glob(f"synthetic_output_*{suffix}"))
    assert len(files) == 2, f"expected two files after the rollover, got: {files}"
    identity = {
        dim_name: np.arange(GLOBAL_SIZES[distributed_dim(dim_name)])
        for dim_name, _ in VARIABLES.values()
    }
    for path, steps in ((files[0], (0, 2)), (files[1], (4,))):
        dataset = (
            xr.open_dataset(path, decode_times=False)
            if backend == common_io.OutputBackend.NETCDF
            else xr.open_zarr(path, decode_times=False, mask_and_scale=False)
        )
        with dataset:
            if mode == common_io.OutputMode.DISTRIBUTED:
                assert (
                    dataset["air_density"].attrs[writers.LAYOUT_ATTRIBUTE]
                    == writers.RANK_BLOCK_LAYOUT
                )
                global_index = {
                    dim_name: dataset[f"{writers.GLOBAL_INDEX_PREFIX}_{dim_name}"].values
                    for dim_name, _ in VARIABLES.values()
                }
            else:
                assert writers.LAYOUT_ATTRIBUTE not in dataset["air_density"].attrs
                global_index = identity
            assert_dataset_is_exact(dataset, steps, global_index)


def test_client_missing_field_closes_the_io_rank_and_raises(tmp_path: pathlib.Path) -> None:
    compute_intercomms, (io_intercomm,) = fake_intercomms(NUM_COMPUTE_RANKS, 1)
    config = io_config(tmp_path, common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED)
    server = ServerThread(io_intercomm)
    clients, infos = make_clients(config, compute_intercomms)
    for client in clients[1:]:
        client.close()
    incomplete = {"air_density": synthetic_state(infos[0], 0)["air_density"]}
    with pytest.raises(errors.IncompleteStateError, match="normal_velocity"):
        clients[0].store(incomplete, dt.datetime(2000, 1, 1))
    server.join()  # the failing client closed, so the IO rank finished cleanly
    with pytest.raises(RuntimeError, match="closed"):
        clients[0].store(synthetic_state(infos[0], 1), dt.datetime(2000, 1, 1, 1))


def test_io_rank_rejects_compute_ranks_at_different_capture_steps(tmp_path: pathlib.Path) -> None:
    compute_intercomms, (io_intercomm,) = fake_intercomms(NUM_COMPUTE_RANKS, 1)
    config = io_config(tmp_path, common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED)
    server = ServerThread(io_intercomm)
    clients, infos = make_clients(config, compute_intercomms)
    clients[0].close()  # closes while the others still ship a capture
    for client, info in zip(clients[1:], infos[1:], strict=True):
        client.store(synthetic_state(info, 0), dt.datetime(2000, 1, 1))
    with pytest.raises(RuntimeError, match="not at the same capture step"):
        server.join()


def test_io_rank_rejects_differing_setups(tmp_path: pathlib.Path) -> None:
    compute_intercomms, (io_intercomm,) = fake_intercomms(NUM_COMPUTE_RANKS, 1)
    config = io_config(tmp_path, common_io.OutputBackend.ZARR, common_io.OutputMode.DISTRIBUTED)
    other_config = dataclasses.replace(config, output_path=tmp_path / "elsewhere")
    server = ServerThread(io_intercomm)
    for rank, intercomm in enumerate(compute_intercomms):
        io_servers.IOClient(
            config=config if rank == 0 else other_config,
            dtime=DTIME,
            grid_id=GRID_ID,
            horizontal_size=horizontal_size(),
            decomposition_info=synthetic_decomposition_info(rank),
            intercomm=intercomm,  # type: ignore[arg-type]  # in-process stand-in
            io_rank=0,
        )
    with pytest.raises(RuntimeError, match="differing output setups"):
        server.join()


def test_io_server_requires_a_served_compute_rank() -> None:
    _, (io_intercomm,) = fake_intercomms(1, 1)
    io_servers.IOServer(io_intercomm, served_ranks=(0,))  # type: ignore[arg-type]  # in-process stand-in
    with pytest.raises(errors.InvalidConfigError, match="serves no compute rank"):
        io_servers.IOServer(io_intercomm, served_ranks=())  # type: ignore[arg-type]  # in-process stand-in
