# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Output through dedicated IO ranks.

The communicator of a run is split into compute ranks and IO ranks
(``decomposition.mpi_decomposition.split_io_ranks``: the last ``IOConfig.io_ranks_per_node``
ranks of every node). The compute ranks run the model and, at every capture step, ship
their owned entries of the output fields to the IO rank serving them (:class:`IOClient`,
the compute side's output monitor); the IO ranks receive them and write
(:class:`IOServer`). Who serves whom is decided by the split
(``mpi_decomposition.assign_io_ranks``): a compute rank ships to an IO rank on its own
node, and the node's compute ranks are shared evenly among its IO ranks.

An IO rank writes through the ordinary output stack, on the IO ranks' communicator:
its local state is the concatenation of the owned entries of the compute ranks it
serves -- a decomposition without halos (``IOServer.receive_setup``) -- so the
distributions of ``io.distributed`` (gathered or rank-block output, per the group's
``mode``) and both writers apply unchanged. With a single IO rank the rank-block store
has one block; the mode still selects the layout.

The compute side is asynchronous: ``store`` strips the halos into a fresh host buffer
per field (``distributed.owned_entries``, as the rank-block distribution does), posts
nonblocking sends of these buffers and returns; the transfer overlaps the following
model steps and completes as the MPI library progresses it. At most
``MAX_IN_FLIGHT_CAPTURES`` capture steps are in flight per compute rank; at that
bound ``store`` blocks until the oldest transfer completed (phase
``io.PHASE_SEND_WAIT`` -- the signal that the IO ranks do not keep up). No thread is
involved on either side.

Protocol (point-to-point on the inter-communicator, addressed by the rank in the other
role's communicator): at construction every compute rank sends its :class:`ClientSetup`
(the output configuration -- the single source of truth, the IO rank checks that its
compute ranks agree --, the grid identity and its owned global indices). At every
capture step it sends a :class:`CaptureHeader` followed by one message per output
variable, in header order; the messages of one source arrive in order, so the IO rank
receives them deterministically without any tags beyond the three message kinds.
``IOClient.close`` completes the transfers and sends an empty header, which ends the
IO rank's ``serve`` loop.

Failure envelope: a compute rank that fails closes its client (the driver closes the
monitor on error paths), so its IO rank finishes and exits cleanly. An IO rank that
fails raises out of ``serve``; the compute ranks cannot observe that -- they block at
the in-flight bound at the latest -- so the caller of ``serve`` must abort the run
(the driver does, ``driver_io.run_io_server``). Every check that must fail on all IO
ranks together (existing output files, mismatching store metadata) is collective on
their communicator through the ordinary output stack.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import logging
import timeit
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import xarray as xr

from icon4py.model.common import exceptions, time
from icon4py.model.common.components import monitor
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base
from icon4py.model.common.io import distributed, io, writers


if TYPE_CHECKING:
    import mpi4py.MPI


log = logging.getLogger(__name__)

#: Upper bound of the capture steps a compute rank keeps in flight to its IO rank. A
#: transfer holds host copies of one capture step's fields, so the bound also caps the
#: staging memory at this many captures. At the bound, ``IOClient.store`` blocks until
#: the oldest transfer completed (phase ``io.PHASE_SEND_WAIT``).
MAX_IN_FLIGHT_CAPTURES: Final[int] = 2

#: Message kinds on the inter-communicator (MPI tags).
_SETUP_TAG: Final[int] = 1
_HEADER_TAG: Final[int] = 2
_DATA_TAG: Final[int] = 3


@dataclasses.dataclass(frozen=True)
class ClientSetup:
    """What a compute rank tells its IO rank once, at construction.

    Attributes:
        config: the output configuration; the IO rank writes with it, so all its
            compute ranks must send an equal one.
        grid_id: identity of the horizontal grid (``uuidOfHGrid``).
        horizontal_size: sizes of the undecomposed global grid.
        owned_global_index: global index of the owned entries per horizontal
            dimension name (``distributed.HORIZONTAL_DIM_NAMES``), the compute rank's
            part of the IO rank's local state.
    """

    config: io.IOConfig
    grid_id: uuid.UUID
    horizontal_size: base.HorizontalGridSize
    owned_global_index: dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class ServerSetup:
    """The setup of an IO rank, assembled from the ``ClientSetup`` of its compute ranks.

    Attributes:
        config: the output configuration (equal on all served compute ranks).
        grid_id: identity of the horizontal grid.
        horizontal_size: sizes of the undecomposed global grid.
        decomposition_info: the IO rank's local state as a decomposition: the owned
            entries of the served compute ranks, concatenated in rank order, all owned
            (no halos).
    """

    config: io.IOConfig
    grid_id: uuid.UUID
    horizontal_size: base.HorizontalGridSize
    decomposition_info: decomposition.DecompositionInfo


@dataclasses.dataclass(frozen=True)
class VariableSpec:
    """Rank-independent description of a shipped output variable.

    Attributes:
        dims: dimension names, horizontal first (as in the model state).
        trailing_shape: shape past the horizontal axis (e.g. the vertical size, empty
            for a purely horizontal field); the horizontal extent is the receiving
            rank's entry count from the setup.
        dtype: element type.
        attrs: the CF/UGRID attributes the writers carry onto the file variable
            (``writers.DATA_VARIABLE_ATTRIBUTES``, those present on the field).
    """

    dims: tuple[str, ...]
    trailing_shape: tuple[int, ...]
    dtype: np.dtype
    attrs: dict[str, str]


@dataclasses.dataclass(frozen=True)
class CaptureHeader:
    """What precedes the data messages of one capture step.

    Attributes:
        model_time: simulation time of the captured state.
        groups: indices into ``IOConfig.field_groups`` of the groups capturing at
            this step.
        variables: the shipped variables in message order (the union of the groups'
            variables); equal on all compute ranks of an IO rank.
    """

    model_time: dt.datetime
    groups: tuple[int, ...]
    variables: dict[str, VariableSpec]


@dataclasses.dataclass
class _Transfer:
    """The pending sends of one capture step and the buffers they read."""

    requests: list[mpi4py.MPI.Request]
    buffers: list[np.ndarray]

    def wait(self) -> None:
        for request in self.requests:
            request.Wait()


class IOClient(monitor.Monitor):
    """Output monitor of a compute rank in a run with dedicated IO ranks.

    Keeps the capture schedules of the field groups (``io.OutputSchedule``); at a
    capture step it strips the halos of the capturing groups' fields and ships them to
    ``io_rank`` (its IO rank in the IO communicator, ``IORankSplit.io_rank``) as
    described in the module docstring. Construction sends the setup, so the IO rank
    must be waiting in ``IOServer.receive_setup``. Like ``IOMonitor``, ``store`` must
    be called at every model step on every compute rank; there is no communication
    among the compute ranks.
    """

    def __init__(
        self,
        *,
        config: io.IOConfig,
        dtime: time.RelativeTime,
        grid_id: uuid.UUID,
        horizontal_size: base.HorizontalGridSize,
        decomposition_info: decomposition.DecompositionInfo,
        intercomm: mpi4py.MPI.Intercomm,
        io_rank: int,
    ):
        self.config = config
        self._intercomm = intercomm
        self._io_rank = io_rank
        self._decomposition_info = decomposition_info.as_host()
        self._schedules = [
            io.OutputSchedule(group.output_interval, dtime) for group in config.field_groups
        ]
        self._in_flight: collections.deque[_Transfer] = collections.deque()
        self._phase_seconds: dict[str, list[float]] = {
            io.PHASE_DISTRIBUTE: [],
            io.PHASE_SEND: [],
            io.PHASE_SEND_WAIT: [],
        }
        self._closed = False
        setup = ClientSetup(
            config=config,
            grid_id=grid_id,
            horizontal_size=horizontal_size,
            owned_global_index={
                dim_name: distributed.owned_global_index(self._decomposition_info, dim)
                for dim, dim_name in distributed.HORIZONTAL_DIM_NAMES.items()
            },
        )
        self._intercomm.send(setup, dest=self._io_rank, tag=_SETUP_TAG)

    @property
    def io_rank(self) -> int:
        """The IO rank this compute rank ships to (rank in the IO communicator)."""
        return self._io_rank

    def store(
        self, state: dict, model_time: dt.datetime, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        """Advance the schedules; at a capture step ship the capturing groups' fields.

        Blocks only when ``MAX_IN_FLIGHT_CAPTURES`` capture steps are still in flight
        (phase "send_wait"). The shipped data is a copy, so the model may mutate the
        state right after this returns.
        """
        groups = tuple(
            index for index, schedule in enumerate(self._schedules) if schedule.at_capture_time()
        )
        for schedule in self._schedules:
            schedule.advance()
        if not groups:
            return
        if self._closed:
            raise RuntimeError("The IO client has been closed; it cannot ship further captures.")
        names = list(
            dict.fromkeys(
                name for index in groups for name in self.config.field_groups[index].variables
            )
        )
        try:
            fields = {name: state[name] for name in names}
        except KeyError as e:
            log.error(f"Field '{e.args[0]}' is missing in state.")
            self.close()
            raise exceptions.IncompleteStateError(e.args[0]) from e

        log.info(f"Shipping fields {names} at {model_time} to IO rank {self._io_rank}")

        start = timeit.default_timer()
        buffers: list[np.ndarray] = []
        variables: dict[str, VariableSpec] = {}
        for name, field in fields.items():
            owned = np.ascontiguousarray(
                distributed.owned_entries(str(field.dims[0]), self._decomposition_info, field)
            )
            buffers.append(owned)
            variables[name] = VariableSpec(
                dims=tuple(str(dim) for dim in field.dims),
                trailing_shape=owned.shape[1:],
                dtype=owned.dtype,
                attrs={
                    key: field.attrs[key]
                    for key in writers.DATA_VARIABLE_ATTRIBUTES
                    if key in field.attrs
                },
            )
        self._phase_seconds[io.PHASE_DISTRIBUTE].append(timeit.default_timer() - start)

        start = timeit.default_timer()
        self._wait_for_slot()
        header = CaptureHeader(model_time=model_time, groups=groups, variables=variables)
        requests = [self._intercomm.isend(header, dest=self._io_rank, tag=_HEADER_TAG)]
        requests.extend(
            self._intercomm.Isend(buffer, dest=self._io_rank, tag=_DATA_TAG) for buffer in buffers
        )
        self._in_flight.append(_Transfer(requests, buffers))
        self._phase_seconds[io.PHASE_SEND].append(timeit.default_timer() - start)

    def _wait_for_slot(self) -> None:
        """Complete the oldest transfers until fewer than the bound are in flight."""
        start = timeit.default_timer()
        while len(self._in_flight) >= MAX_IN_FLIGHT_CAPTURES:
            self._in_flight.popleft().wait()
        self._phase_seconds[io.PHASE_SEND_WAIT].append(timeit.default_timer() - start)

    def variables_at_capture_time(self) -> set[str]:
        """The variables some field group captures at the current ``store`` call.

        Pure and identical on all compute ranks, like ``IOMonitor.variables_at_capture_time``.
        """
        return {
            name
            for index, schedule in enumerate(self._schedules)
            if schedule.at_capture_time()
            for name in self.config.field_groups[index].variables
        }

    @property
    def phase_seconds(self) -> dict[str, list[float]]:
        """Per-phase wall-clock seconds of every capture step (see ``io.PHASE_*``)."""
        return dict(self._phase_seconds)

    def report_timings(self) -> None:
        """Log this rank's accumulated output overhead per phase (communication-free)."""
        io.log_phase_timings(
            f"IO client of compute rank {self._intercomm.Get_rank()}",
            self.phase_seconds,
            self._intercomm.Get_rank(),
        )

    def close(self) -> None:
        """Complete the in-flight transfers and end the IO rank's serve loop; idempotent.

        Final: a subsequent capture step raises.
        """
        if self._closed:
            return
        self._closed = True
        while self._in_flight:
            self._in_flight.popleft().wait()
        self._intercomm.send(None, dest=self._io_rank, tag=_HEADER_TAG)


class IOServer:
    """The receiving side of a dedicated IO rank.

    ``receive_setup`` collects the setups of the served compute ranks (``served_ranks``,
    in the compute communicator, see ``IORankSplit.served_compute_ranks``) and
    describes the monitor to build from them; ``serve`` then feeds every received
    capture step to that monitor until the compute ranks close (see the module
    docstring).
    """

    def __init__(self, intercomm: mpi4py.MPI.Intercomm, served_ranks: Sequence[int]):
        self._intercomm = intercomm
        self._served_ranks = tuple(served_ranks)
        if not self._served_ranks:
            raise exceptions.InvalidConfigError(
                f"IO rank {intercomm.Get_rank()} serves no compute rank: every IO rank "
                f"needs at least one."
            )
        #: per horizontal dimension name, the owned entry count of every served rank
        self._entry_counts: dict[str, list[int]] = {}
        #: receive buffers per variable, allocated at first use and reused
        self._buffers: dict[str, np.ndarray] = {}

    @property
    def served_ranks(self) -> tuple[int, ...]:
        """The compute ranks (in the compute communicator) shipping to this IO rank."""
        return self._served_ranks

    def receive_setup(self) -> ServerSetup:
        """Receive the ``ClientSetup`` of every served compute rank; blocks until all sent.

        Raises:
            RuntimeError: if the served compute ranks disagree on the configuration
                or the grid.
        """
        setups: list[ClientSetup] = [
            self._intercomm.recv(source=rank, tag=_SETUP_TAG) for rank in self._served_ranks
        ]
        first = setups[0]
        for rank, setup in zip(self._served_ranks, setups, strict=True):
            if (
                setup.config != first.config
                or setup.grid_id != first.grid_id
                or setup.horizontal_size != first.horizontal_size
            ):
                raise RuntimeError(
                    f"Compute ranks {self._served_ranks[0]} and {rank} sent differing "
                    f"output setups (configuration or grid) to IO rank "
                    f"{self._intercomm.Get_rank()}."
                )
        info = decomposition.DecompositionInfo()
        for dim, dim_name in distributed.HORIZONTAL_DIM_NAMES.items():
            parts = [setup.owned_global_index[dim_name] for setup in setups]
            global_index = np.concatenate(parts).astype(np.int64, copy=False)
            info.set_dimension(
                dim,
                global_index,
                np.ones(global_index.shape[0], dtype=bool),
                np.full(
                    global_index.shape[0],
                    decomposition.DecompositionFlag.OWNED.value,
                    dtype=np.int32,
                ),
            )
            self._entry_counts[dim_name] = [part.shape[0] for part in parts]
        return ServerSetup(
            config=first.config,
            grid_id=first.grid_id,
            horizontal_size=first.horizontal_size,
            decomposition_info=info,
        )

    def serve(self, io_monitor: io.IOMonitor) -> None:
        """Receive capture steps and write them until every served compute rank closed.

        ``io_monitor`` must be built from ``receive_setup`` (its process properties are
        the IO ranks'); it is closed when this returns, also on failure -- an IO-side
        error propagates and the caller must abort the run (module docstring).
        """
        try:
            while (header := self._receive_header()) is not None:
                state = self._receive_state(header)
                io_monitor.capture(state, header.model_time, header.groups)
            io_monitor.report_timings()
        finally:
            io_monitor.close()

    def _receive_header(self) -> CaptureHeader | None:
        """The next capture header, None once the compute ranks closed.

        Raises:
            RuntimeError: if the served ranks are not at the same capture step.
        """
        headers: list[CaptureHeader | None] = [
            self._intercomm.recv(source=rank, tag=_HEADER_TAG) for rank in self._served_ranks
        ]
        first = headers[0]
        for rank, header in zip(self._served_ranks, headers, strict=True):
            if header != first:
                raise RuntimeError(
                    f"Compute ranks {self._served_ranks[0]} and {rank} sent differing capture "
                    f"headers to IO rank {self._intercomm.Get_rank()}: the compute ranks are "
                    f"not at the same capture step."
                )
        return first

    def _receive_state(self, header: CaptureHeader) -> dict[str, xr.DataArray]:
        """Receive the data messages of a capture step into the reused buffers.

        Each variable's buffer holds the served ranks' entries concatenated in rank
        order, matching the decomposition info of the setup.
        """
        state: dict[str, xr.DataArray] = {}
        for name, spec in header.variables.items():
            counts = self._entry_counts[spec.dims[0]]
            buffer = self._buffer(name, (sum(counts), *spec.trailing_shape), spec.dtype)
            offset = 0
            for rank, count in zip(self._served_ranks, counts, strict=True):
                self._intercomm.Recv(buffer[offset : offset + count], source=rank, tag=_DATA_TAG)
                offset += count
            state[name] = xr.DataArray(buffer, dims=spec.dims, attrs=dict(spec.attrs))
        return state

    def _buffer(self, name: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        buffer = self._buffers.get(name)
        if buffer is None or buffer.shape != shape or buffer.dtype != dtype:
            buffer = np.empty(shape, dtype=dtype)
            self._buffers[name] = buffer
        return buffer
