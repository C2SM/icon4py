# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The dedicated IO ranks of a run: which ranks write, and the communicators of the two roles.

Every run has its :class:`RankRoles`, built by ``create_rank_roles`` from the output
configuration: :class:`NoIORanks` when the ranks running the model write themselves,
an :class:`IORankSplit` when some ranks are dedicated to output.

A run with dedicated IO ranks (``IOConfig.io_ranks``) sets them up in three steps:
``allgather_node_ids`` finds the node of every rank of the run's communicator,
``assign_io_ranks`` chooses the IO ranks and the compute ranks each of them serves
(:class:`IORankAssignment`, a function of the nodes alone), and
``split_io_ranks`` builds the communicator of either role and the inter-communicator
between them (:class:`IORankSplit`). The protocol the two roles speak is in
``io_servers``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import signal
from collections.abc import Collection, Iterator, Sequence
from typing import TYPE_CHECKING, Final, Protocol

from icon4py.model.common import exceptions
from icon4py.model.common.decomposition import definitions as decomposition, mpi_decomposition
from icon4py.model.common.io import io


if TYPE_CHECKING:
    import mpi4py.MPI


log = logging.getLogger(__name__)

#: Names of the role communicators of ``split_io_ranks``, shown in the rank prefix of
#: distributed log records (``ProcessProperties.comm_name``).
COMPUTE_COMM_NAME: Final[str] = "compute"
IO_COMM_NAME: Final[str] = "io"

#: Seconds a rank may live on after it called ``MPI_Abort`` (see ``_abort``).
_ABORT_GRACE_SECONDS: Final[int] = 10


def _balanced_blocks(items: Sequence[int], num_blocks: int) -> list[tuple[int, ...]]:
    """Split ``items`` into ``num_blocks`` contiguous blocks whose sizes differ by at most one.

    The first ``len(items) % num_blocks`` blocks are the larger ones.
    """
    size, extra = divmod(len(items), num_blocks)
    blocks: list[tuple[int, ...]] = []
    start = 0
    for index in range(num_blocks):
        stop = start + size + (1 if index < extra else 0)
        blocks.append(tuple(items[start:stop]))
        start = stop
    return blocks


def _ranks_of_nodes(node_ids: Sequence[int]) -> list[list[int]]:
    """The ranks of every node in increasing order, the nodes ordered by their lowest rank."""
    ranks_of_node: dict[int, list[int]] = {}
    for rank, node in enumerate(node_ids):
        ranks_of_node.setdefault(node, []).append(rank)
    return list(ranks_of_node.values())


@dataclasses.dataclass(frozen=True)
class IORankAssignment:
    """The dedicated IO ranks of a communicator and the compute ranks each one serves.

    ``compute_ranks`` and ``io_ranks`` hold the ranks of the split communicator in
    increasing order. ``split_io_ranks`` builds the role communicators in that order,
    so a rank's position in its tuple is its rank in the role communicator;
    ``io_rank_of`` and ``served_compute_ranks`` speak in those role ranks.

    Attributes:
        node_ids: the node of every rank of the split communicator, as given to the
            ``assign_*`` function (see ``allgather_node_ids``).
        compute_ranks: the compute ranks, as ranks of the split communicator.
        io_ranks: the IO ranks, as ranks of the split communicator.
        io_rank_of_compute_rank: per compute rank (compute-communicator numbering), the
            IO rank (IO-communicator numbering) it ships to.
    """

    node_ids: tuple[int, ...]
    compute_ranks: tuple[int, ...]
    io_ranks: tuple[int, ...]
    io_rank_of_compute_rank: tuple[int, ...]

    def io_rank_of(self, compute_rank: int) -> int:
        """The IO rank (IO communicator) serving a compute rank (compute communicator)."""
        return self.io_rank_of_compute_rank[compute_rank]

    def served_compute_ranks(self, io_rank: int) -> tuple[int, ...]:
        """The compute ranks (compute communicator) served by an IO rank (IO communicator)."""
        return tuple(
            compute_rank
            for compute_rank, served_by in enumerate(self.io_rank_of_compute_rank)
            if served_by == io_rank
        )


def _serve_compute_ranks(node_ids: Sequence[int], io_ranks: Collection[int]) -> IORankAssignment:
    """Share the compute ranks, all ranks but the chosen ``io_ranks``, among the IO ranks.

    The compute ranks of a node with IO ranks are shared evenly among these, in rank
    order, so they ship on their node. The compute ranks of the nodes without an IO
    rank are then handed out node by node, in contiguous blocks sized to level the
    number of compute ranks per IO rank.

    Raises:
        InvalidConfigError: if an IO rank is left without a compute rank.
    """
    chosen = frozenset(io_ranks)
    served: dict[int, list[int]] = {io_rank: [] for io_rank in sorted(chosen)}
    off_node: list[int] = []
    for ranks in _ranks_of_nodes(node_ids):
        node_io_ranks = [rank for rank in ranks if rank in chosen]
        node_compute_ranks = [rank for rank in ranks if rank not in chosen]
        if not node_io_ranks:
            off_node.extend(node_compute_ranks)
            continue
        blocks = _balanced_blocks(node_compute_ranks, len(node_io_ranks))
        for io_rank, block in zip(node_io_ranks, blocks, strict=True):
            served[io_rank].extend(block)
    # every off-node compute rank counts for the IO rank serving the fewest so far (the
    # lowest one on ties); the ranks are then handed out in blocks of these sizes
    num_served = {io_rank: len(ranks) for io_rank, ranks in served.items()}
    for _ in off_node:
        num_served[min(num_served, key=num_served.__getitem__)] += 1
    start = 0
    for io_rank, ranks in served.items():
        stop = start + num_served[io_rank] - len(ranks)
        ranks.extend(off_node[start:stop])
        start = stop
    for io_rank, ranks in served.items():
        if not ranks:
            raise exceptions.InvalidConfigError(
                f"Rank {io_rank} would be an IO rank without a compute rank to serve: reduce "
                f"the number of IO ranks."
            )
    compute_ranks = sorted(rank for ranks in served.values() for rank in ranks)
    io_rank_of = {
        compute_rank: index for index, ranks in enumerate(served.values()) for compute_rank in ranks
    }
    return IORankAssignment(
        node_ids=tuple(node_ids),
        compute_ranks=tuple(compute_ranks),
        io_ranks=tuple(served),
        io_rank_of_compute_rank=tuple(io_rank_of[rank] for rank in compute_ranks),
    )


def assign_io_ranks_per_node(node_ids: Sequence[int], io_ranks_per_node: int) -> IORankAssignment:
    """Choose the dedicated IO ranks: the last ``io_ranks_per_node`` ranks of every node.

    ``node_ids[rank]`` identifies the node of every rank of the communicator (see
    ``allgather_node_ids``). The other ranks of a node are its compute ranks; in rank
    order, they are shared evenly among the node's IO ranks. A compute rank thus
    always ships to an IO rank on its own node, whatever the placement of the ranks on
    the nodes, and the transfer never crosses the network.

    Raises:
        InvalidConfigError: if ``io_ranks_per_node`` is not positive, or a node has
            fewer compute ranks than IO ranks (every IO rank serves at least one).
    """
    if io_ranks_per_node < 1:
        raise exceptions.InvalidConfigError(
            f"Invalid number of IO ranks per node {io_ranks_per_node}: must be at least 1."
        )
    io_ranks: list[int] = []
    for ranks in _ranks_of_nodes(node_ids):
        if len(ranks) < 2 * io_ranks_per_node:
            raise exceptions.InvalidConfigError(
                f"The node of rank {ranks[0]} has {len(ranks)} ranks: with {io_ranks_per_node} "
                f"IO rank(s) per node it needs at least {2 * io_ranks_per_node} (every IO rank "
                f"serves at least one compute rank)."
            )
        io_ranks.extend(ranks[-io_ranks_per_node:])
    return _serve_compute_ranks(node_ids, io_ranks)


def assign_io_ranks_on_fullest_nodes(
    node_ids: Sequence[int], num_io_ranks: int
) -> IORankAssignment:
    """Choose ``num_io_ranks`` dedicated IO ranks in total: the last ranks of the fullest nodes.

    The IO ranks are taken one at a time, always the last remaining rank of the
    fullest node; when fewer IO ranks are left than there are equally full nodes, they
    are spread evenly over these, in node order. A run launched with ``num_io_ranks``
    tasks more than compute ranks, the surplus tasks on nodes that also hold compute
    ranks (where a launcher filling the nodes in order puts them), thus turns exactly
    the surplus tasks into IO ranks, and every node keeps its compute ranks. Surplus
    tasks on nodes of their own are not recognized: the IO ranks are taken from the
    fullest nodes, whatever their tasks were meant for (a single surplus task on a
    node of its own computes, and the IO rank is taken from another node). A compute
    rank ships to an IO rank on its own node if there is one, otherwise across nodes
    (see ``_serve_compute_ranks``). ``node_ids`` as in ``assign_io_ranks_per_node``.

    Raises:
        InvalidConfigError: if ``num_io_ranks`` is not positive, or the run has fewer
            compute ranks than IO ranks, or an IO rank is left without a compute rank.
    """
    if num_io_ranks < 1:
        raise exceptions.InvalidConfigError(
            f"Invalid number of IO ranks {num_io_ranks}: must be at least 1."
        )
    if len(node_ids) < 2 * num_io_ranks:
        raise exceptions.InvalidConfigError(
            f"The run has {len(node_ids)} ranks: with {num_io_ranks} IO rank(s) it needs at "
            f"least {2 * num_io_ranks} (every IO rank serves at least one compute rank)."
        )
    ranks_of_nodes = _ranks_of_nodes(node_ids)
    num_compute_ranks = [len(ranks) for ranks in ranks_of_nodes]
    remaining = num_io_ranks
    while remaining > 0:
        most = max(num_compute_ranks)
        fullest = [node for node, count in enumerate(num_compute_ranks) if count == most]
        if len(fullest) > remaining:
            # fewer IO ranks left than equally full nodes: spread them evenly over these
            fullest = [fullest[index * len(fullest) // remaining] for index in range(remaining)]
        for node in fullest:
            num_compute_ranks[node] -= 1
        remaining -= len(fullest)
    io_ranks = [
        rank
        for ranks, count in zip(ranks_of_nodes, num_compute_ranks, strict=True)
        for rank in ranks[count:]
    ]
    return _serve_compute_ranks(node_ids, io_ranks)


def assign_io_ranks(io_ranks: io.IORanks, node_ids: Sequence[int]) -> IORankAssignment:
    """Choose the dedicated IO ranks an output configuration asks for (``IOConfig.io_ranks``).

    ``node_ids[rank]`` identifies the node of every rank of the communicator to split
    (``allgather_node_ids``); the result is what ``split_io_ranks`` splits it by.
    ``IORanksPerNode`` puts IO ranks on every node; ``IORanksInTotal`` gives their
    total number, taken from the nodes holding the most ranks.

    Raises:
        InvalidConfigError: if the run has too few ranks for the IO ranks (see the
            ``assign_*`` functions).
    """
    match io_ranks:
        case io.IORanksPerNode(count=count):
            return assign_io_ranks_per_node(node_ids, count)
        case io.IORanksInTotal(count=count):
            return assign_io_ranks_on_fullest_nodes(node_ids, count)


def num_ranks_per_node(assignment: IORankAssignment) -> list[tuple[int, int]]:
    """The numbers of compute ranks and of IO ranks of every node.

    One ``(compute, IO)`` pair per node, the nodes ordered by their lowest rank.
    """
    io_ranks = frozenset(assignment.io_ranks)
    return [
        (sum(rank not in io_ranks for rank in ranks), sum(rank in io_ranks for rank in ranks))
        for ranks in _ranks_of_nodes(assignment.node_ids)
    ]


def describe_layout(assignment: IORankAssignment) -> str:
    """Where the ranks of the two roles run, in one line.

    Lists the compute and IO ranks of every node in node order, runs of equal nodes
    counted once: ``4+1 (x2), 4+0 (x8)`` reads four compute ranks and one IO rank on
    each of the first two nodes, four compute ranks and no IO rank on the other eight.
    """
    runs: list[tuple[tuple[int, int], int]] = []
    for counts in num_ranks_per_node(assignment):
        if runs and runs[-1][0] == counts:
            runs[-1] = (counts, runs[-1][1] + 1)
        else:
            runs.append((counts, 1))
    per_node = ", ".join(
        f"{num_compute}+{num_io}" + (f" (x{num_nodes})" if num_nodes > 1 else "")
        for (num_compute, num_io), num_nodes in runs
    )
    num_nodes = sum(num_nodes for _, num_nodes in runs)
    return (
        f"{len(assignment.compute_ranks)} compute rank(s) and {len(assignment.io_ranks)} IO "
        f"rank(s) on {num_nodes} node(s); compute+IO ranks per node: {per_node}"
    )


def log_layout(assignment: IORankAssignment) -> None:
    """Log where the ranks of the two roles run; warn if the nodes differ in compute ranks.

    The IO ranks are taken from the tasks the launcher provides, so a launch that does
    not match the configuration still runs: one IO rank too many takes a compute rank
    from some node, whose accelerator then idles. Nodes holding unequal numbers of
    compute ranks are the sign of it, hence the warning. A launch that is short of the
    same number of tasks on every node (e.g. no task for the IO rank at all) cannot be
    told from an intended one: only the logged layout shows it. To be called on one rank.
    """
    layout = describe_layout(assignment)
    log.info(f"Dedicated IO ranks: {layout}.")
    num_compute_ranks = [num_compute for num_compute, _ in num_ranks_per_node(assignment)]
    if min(num_compute_ranks) != max(num_compute_ranks):
        log.warning(
            f"The nodes hold unequal numbers of compute ranks ({layout}): check that the "
            f"launched tasks match the dedicated IO ranks of the configuration ('io_ranks')."
        )


def allgather_node_ids(process_props: decomposition.ProcessProperties) -> list[int]:
    """The node of every rank of the communicator, identified by the lowest rank on it.

    The nodes are found through MPI's shared-memory split (``MPI_COMM_TYPE_SHARED``):
    the ranks that can share memory are on one node. Collective on the communicator;
    works on a single-rank communicator without an MPI installation.
    """
    if process_props.is_single_rank():
        return [0]
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

    comm = process_props.comm
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=process_props.rank)
    try:
        node_id: int = node_comm.bcast(process_props.rank, root=0)
    finally:
        node_comm.Free()
    return comm.allgather(node_id)


def _abort(comm: mpi4py.MPI.Comm) -> None:
    """Abort the ranks of ``comm``, with an alarm that ends this process if the abort does not.

    ``MPI_Abort`` alone is not enough: it ends the process through ``exit``, and an
    exit handler can block. With a parallel netCDF file open, HDF5's handler closes the
    file, which is collective (a barrier) on ranks that never join: a dedicated IO rank
    that failed alone called the abort and the job hung all the same (Cray MPICH 9.1
    under Slurm, which ends the job only once the rank has exited; 2026-11). The
    handler runs with the interpreter lock held, so no Python thread could step in. An
    alarm therefore ends the process if the abort has not done so in time, and the
    launcher ends the job for the rank that was killed.

    The alarm relies on the default action of ``SIGALRM`` (terminate the process): a
    Python-level handler for it could not run either. It stays armed when the abort
    returns or raises, since the rank then heads for MPI's shutdown, where it would
    wait for the other ranks.
    """
    signal.alarm(_ABORT_GRACE_SECONDS)
    comm.Abort(1)


class RankRoles(Protocol):
    """The role of this rank in a run and the communicator the role works on.

    Whether or not a run dedicates ranks to output, a rank asks the same questions of
    its roles: with dedicated IO ranks the answers come from an :class:`IORankSplit`,
    without from :class:`NoIORanks`.
    """

    @property
    def is_io_rank(self) -> bool:
        """Whether this rank only writes output; never so without dedicated IO ranks."""
        ...

    @property
    def process_props(self) -> decomposition.ProcessProperties:
        """Process properties of the communicator this rank's role works on."""
        ...

    def report_layout(self) -> None:
        """Log where the ranks of the roles run, on one rank of the run."""
        ...

    def abort_on_failure(self) -> contextlib.AbstractContextManager[None]:
        """Context in which a failure of this rank must not leave other ranks waiting."""
        ...


@dataclasses.dataclass(frozen=True)
class NoIORanks:
    """The roles of a run without dedicated IO ranks: every rank runs the model and writes.

    Attributes:
        process_props: process properties of the run's communicator.
    """

    process_props: decomposition.ProcessProperties

    @property
    def is_io_rank(self) -> bool:
        return False

    def report_layout(self) -> None:
        """Nothing to report: all ranks have the same role."""

    @contextlib.contextmanager
    def abort_on_failure(self) -> Iterator[None]:
        """A failure propagates unchanged: no rank waits for another role."""
        yield


@dataclasses.dataclass(frozen=True)
class IORankSplit:
    """A communicator split into compute ranks and dedicated IO ranks.

    Every rank belongs to exactly one role; the roles talk through an
    inter-communicator (``io.io_servers``: compute ranks ship their output fields to
    the IO ranks, which write them). Everything else the model does -- halo exchanges,
    reductions, the grid decomposition -- runs on the compute role's communicator, i.e.
    on the ``process_props`` of a compute rank; nothing but the output layer knows the
    IO ranks exist.

    Attributes:
        run_props: process properties of the communicator that was split. After the
            split it serves a single purpose, aborting the ranks of both roles together.
        assignment: the choice of IO ranks the communicator was split by.
        is_io_rank: whether this rank is a dedicated IO rank.
        process_props: process properties of this rank's role communicator (compute or
            IO); named ``COMPUTE_COMM_NAME`` / ``IO_COMM_NAME``.
        intercomm: inter-communicator whose remote group is the other role. Ranks are
            addressed by their rank in that role's communicator.
        io_rank: on a compute rank, the IO rank (in the IO communicator) it ships to;
            None on an IO rank.
        served_compute_ranks: on an IO rank, the compute ranks (in the compute
            communicator) it serves, in increasing order; empty on a compute rank.
    """

    run_props: decomposition.ProcessProperties
    assignment: IORankAssignment
    is_io_rank: bool
    process_props: mpi_decomposition.MPICommProcessProperties
    intercomm: mpi4py.MPI.Intercomm
    io_rank: int | None
    served_compute_ranks: tuple[int, ...]

    def report_layout(self) -> None:
        """Log the compute and IO ranks of every node, on rank 0 of the compute ranks."""
        if not self.is_io_rank and self.process_props.rank == 0:
            log_layout(self.assignment)

    @contextlib.contextmanager
    def abort_on_failure(self) -> Iterator[None]:
        """Abort the whole run when this rank fails.

        Neither role can observe the other's failure: an IO rank waits for the setup or
        the next capture of its compute ranks, and compute ranks block at the in-flight
        bound once their IO rank stopped receiving. The failing rank would then wait in
        MPI's shutdown for ranks that never get there, and the job would hang instead
        of failing. The abort may interrupt IO ranks writing their last captures, and
        it waits for this rank's own exit handlers for a few seconds only (``_abort``).
        """
        try:
            yield
        except BaseException:
            role = "IO" if self.is_io_rank else "Compute"
            log.exception(f"{role} rank {self.process_props.rank} failed; aborting the run.")
            _abort(self.run_props.comm)
            raise


def split_io_ranks(
    process_props: decomposition.ProcessProperties, assignment: IORankAssignment
) -> IORankSplit:
    """Split the communicator into the compute ranks and the IO ranks of ``assignment``.

    ``assignment`` partitions the ranks of the communicator of ``process_props``: it
    is ``assign_io_ranks`` (or one of the ``assign_*`` functions) applied to the
    communicator's ``allgather_node_ids``. Collective on that communicator.

    Raises:
        InvalidConfigError: on a single-node run (no MPI communicator).
    """
    if process_props.comm is None:
        raise exceptions.InvalidConfigError(
            "Dedicated IO ranks require an MPI run: the process properties carry no communicator."
        )
    is_io_rank = process_props.rank in assignment.io_ranks
    # the role communicators keep the order of the split communicator
    role_comm = process_props.comm.Split(color=int(is_io_rank), key=process_props.rank)
    role_comm.Set_name(IO_COMM_NAME if is_io_rank else COMPUTE_COMM_NAME)
    # the leaders are rank 0 of each role: the lowest compute rank and the lowest IO
    # rank of the split communicator
    remote_leader = assignment.compute_ranks[0] if is_io_rank else assignment.io_ranks[0]
    intercomm = role_comm.Create_intercomm(
        local_leader=0, peer_comm=process_props.comm, remote_leader=remote_leader, tag=0
    )
    role_props = mpi_decomposition.MPICommProcessProperties(role_comm)
    return IORankSplit(
        run_props=process_props,
        assignment=assignment,
        is_io_rank=is_io_rank,
        process_props=role_props,
        intercomm=intercomm,
        io_rank=None if is_io_rank else assignment.io_rank_of(role_props.rank),
        served_compute_ranks=assignment.served_compute_ranks(role_props.rank) if is_io_rank else (),
    )


def create_rank_roles(
    io_config: io.IOConfig | None, process_props: decomposition.ProcessProperties
) -> RankRoles:
    """The roles of the ranks of a run, from its output configuration.

    Without output, or without dedicated IO ranks in it (``IOConfig.io_ranks``), every
    rank runs the model on the communicator of ``process_props``. Otherwise that
    communicator is split into compute ranks and IO ranks (collective on it).
    """
    if io_config is None or io_config.io_ranks is None:
        return NoIORanks(process_props)
    assignment = assign_io_ranks(io_config.io_ranks, allgather_node_ids(process_props))
    return split_io_ranks(process_props, assignment)
