# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""MPI-free tests of the choice of dedicated IO ranks (``io.io_ranks``)."""

import collections
import logging
import signal
from collections.abc import Callable
from unittest import mock

import pytest

import icon4py.model.common.exceptions as errors
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.io import io as common_io, io_ranks


def block_layout(ranks_per_node: int, num_nodes: int) -> list[int]:
    """Consecutive ranks share a node: the launchers' default placement."""
    return [rank // ranks_per_node for rank in range(ranks_per_node * num_nodes)]


def cyclic_layout(ranks_per_node: int, num_nodes: int) -> list[int]:
    """Consecutive ranks go to different nodes (``--distribution=cyclic``)."""
    return [rank % num_nodes for rank in range(ranks_per_node * num_nodes)]


def served_ranks(assignment: io_ranks.IORankAssignment) -> dict[int, list[int]]:
    """The compute ranks every IO rank serves, all as ranks of the split communicator."""
    return {
        rank: [
            assignment.compute_ranks[compute_rank]
            for compute_rank in assignment.served_compute_ranks(io_rank)
        ]
        for io_rank, rank in enumerate(assignment.io_ranks)
    }


# --------------------------------------------------------------------------------------
# IO ranks on every node
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_ids",
    [block_layout(5, 2), cyclic_layout(5, 2), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]],
    ids=["block", "cyclic", "uneven"],
)
@pytest.mark.parametrize("io_ranks_per_node", [1, 2])
def test_assign_io_ranks_puts_io_ranks_on_every_node(
    node_ids: list[int], io_ranks_per_node: int
) -> None:
    assignment = io_ranks.assign_io_ranks_per_node(node_ids, io_ranks_per_node)
    ranks_of_node = collections.defaultdict(list)
    for rank, node in enumerate(node_ids):
        ranks_of_node[node].append(rank)

    # the last ranks of every node are its IO ranks; with the compute ranks they partition
    # the communicator, and rank 0 computes
    expected_io_ranks = sorted(
        rank for ranks in ranks_of_node.values() for rank in ranks[-io_ranks_per_node:]
    )
    assert list(assignment.io_ranks) == expected_io_ranks
    assert sorted(assignment.compute_ranks + assignment.io_ranks) == list(range(len(node_ids)))
    assert assignment.compute_ranks[0] == 0

    # every compute rank ships to an IO rank on its own node, and serving is the inverse
    for compute_rank, rank in enumerate(assignment.compute_ranks):
        io_rank = assignment.io_rank_of(compute_rank)
        assert node_ids[assignment.io_ranks[io_rank]] == node_ids[rank]
        assert compute_rank in assignment.served_compute_ranks(io_rank)
    served = [
        compute_rank
        for io_rank in range(len(assignment.io_ranks))
        for compute_rank in assignment.served_compute_ranks(io_rank)
    ]
    assert sorted(served) == list(range(len(assignment.compute_ranks)))

    # the compute ranks of a node are shared evenly among its IO ranks
    for ranks in ranks_of_node.values():
        sizes = {
            len(assignment.served_compute_ranks(assignment.io_ranks.index(rank)))
            for rank in ranks[-io_ranks_per_node:]
        }
        assert min(sizes) >= 1 and max(sizes) - min(sizes) <= 1


def test_assign_io_ranks_single_node_takes_the_last_ranks() -> None:
    assignment = io_ranks.assign_io_ranks_per_node([0] * 6, 2)
    assert assignment.io_ranks == (4, 5)
    assert assignment.compute_ranks == (0, 1, 2, 3)
    assert [assignment.io_rank_of(compute_rank) for compute_rank in range(4)] == [0, 0, 1, 1]
    assert assignment.served_compute_ranks(0) == (0, 1)
    assert assignment.served_compute_ranks(1) == (2, 3)


def test_assign_io_ranks_rejects_nodes_short_of_compute_ranks() -> None:
    # the node of rank 4 has two ranks: two IO ranks would leave it no compute rank
    with pytest.raises(errors.InvalidConfigError, match="node of rank 4 has 2 ranks"):
        io_ranks.assign_io_ranks_per_node([0, 0, 0, 0, 1, 1], 2)
    with pytest.raises(errors.InvalidConfigError, match="at least 1"):
        io_ranks.assign_io_ranks_per_node([0, 0], 0)


# --------------------------------------------------------------------------------------
# A total number of IO ranks, on the fullest nodes
# --------------------------------------------------------------------------------------


def test_fullest_nodes_turn_the_surplus_tasks_into_io_ranks() -> None:
    # ten nodes with four compute ranks each and two tasks more, which the launcher
    # put on the first two nodes
    node_ids = [0] * 5 + [1] * 5 + [node for node in range(2, 10) for _ in range(4)]
    assignment = io_ranks.assign_io_ranks_on_fullest_nodes(node_ids, 2)

    # the fifth task of the two fuller nodes writes: every node keeps four compute ranks
    assert assignment.io_ranks == (4, 9)
    compute_ranks_per_node = collections.Counter(
        node_ids[rank] for rank in assignment.compute_ranks
    )
    assert compute_ranks_per_node == dict.fromkeys(range(10), 4)
    # a compute rank ships on its node where that has an IO rank; the other nodes are
    # handed out in contiguous blocks that level the IO ranks
    assert served_ranks(assignment) == {
        4: [0, 1, 2, 3, *range(10, 26)],
        9: [5, 6, 7, 8, *range(26, 42)],
    }


def test_fullest_nodes_are_found_wherever_the_surplus_tasks_are() -> None:
    # four nodes with cyclically placed ranks; the nodes 1 and 3 hold a rank more
    node_ids = [0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    assignment = io_ranks.assign_io_ranks_on_fullest_nodes(node_ids, 2)
    assert assignment.io_ranks == (8, 9)
    assert served_ranks(assignment) == {8: [0, 1, 4, 5], 9: [2, 3, 6, 7]}


def test_fullest_nodes_spread_the_io_ranks_over_equally_full_nodes() -> None:
    # no node holds more ranks than the others: the nodes 0 and 5 give their last rank
    assignment = io_ranks.assign_io_ranks_on_fullest_nodes(block_layout(4, 10), 2)
    assert assignment.io_ranks == (3, 23)
    assert served_ranks(assignment) == {
        3: [0, 1, 2, *range(4, 20)],
        23: [20, 21, 22, *range(24, 40)],
    }


@pytest.mark.parametrize(
    "node_ids, io_ranks_per_node",
    [(block_layout(5, 3), 1), (cyclic_layout(6, 2), 2), ([0] * 6, 2)],
    ids=["block", "cyclic", "single-node"],
)
def test_fullest_nodes_with_the_same_surplus_on_every_node_are_the_per_node_choice(
    node_ids: list[int], io_ranks_per_node: int
) -> None:
    num_io_ranks = io_ranks_per_node * len(set(node_ids))
    assert io_ranks.assign_io_ranks_on_fullest_nodes(
        node_ids, num_io_ranks
    ) == io_ranks.assign_io_ranks_per_node(node_ids, io_ranks_per_node)


def test_fullest_nodes_reject_runs_short_of_compute_ranks() -> None:
    with pytest.raises(errors.InvalidConfigError, match="at least 1"):
        io_ranks.assign_io_ranks_on_fullest_nodes([0, 0], 0)
    # three IO ranks need three compute ranks
    with pytest.raises(errors.InvalidConfigError, match="run has 5 ranks"):
        io_ranks.assign_io_ranks_on_fullest_nodes([0, 0, 0, 1, 1], 3)


def test_io_rank_without_a_compute_rank_is_rejected() -> None:
    # two nodes with three ranks each: the first is left with one compute rank for its
    # two IO ranks, and the compute ranks of the other ship to its own IO rank
    with pytest.raises(errors.InvalidConfigError, match="Rank 2 would be an IO rank without"):
        io_ranks.assign_io_ranks_on_fullest_nodes([0, 0, 0, 1, 1, 1], 3)


# --------------------------------------------------------------------------------------
# Both choices
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "assign",
    [
        io_ranks.assign_io_ranks_per_node,
        io_ranks.assign_io_ranks_on_fullest_nodes,
    ],
)
@pytest.mark.parametrize(
    "node_ids",
    [block_layout(6, 3), cyclic_layout(6, 3), [0] * 7 + [1] * 6 + [2] * 6 + [3] * 7],
    ids=["block", "cyclic", "uneven"],
)
def test_every_choice_partitions_the_ranks_and_serves_all_compute_ranks(
    assign: Callable[[list[int], int], io_ranks.IORankAssignment], node_ids: list[int]
) -> None:
    assignment = assign(node_ids, 2)
    assert sorted(assignment.compute_ranks + assignment.io_ranks) == list(range(len(node_ids)))
    served = served_ranks(assignment)
    assert all(served.values())
    assert sorted(rank for ranks in served.values() for rank in ranks) == list(
        assignment.compute_ranks
    )
    for compute_rank, rank in enumerate(assignment.compute_ranks):
        assert rank in served[assignment.io_ranks[assignment.io_rank_of(compute_rank)]]


def test_node_ids_of_a_single_rank_run_need_no_mpi() -> None:
    assert io_ranks.allgather_node_ids(decomp_defs.SingleNodeProcessProperties()) == [0]


def test_split_io_ranks_requires_an_mpi_run() -> None:
    assignment = io_ranks.assign_io_ranks_per_node([0, 0], 1)
    with pytest.raises(errors.InvalidConfigError, match="require an MPI run"):
        io_ranks.split_io_ranks(decomp_defs.SingleNodeProcessProperties(), assignment)


# --------------------------------------------------------------------------------------
# the choice an output configuration asks for
# --------------------------------------------------------------------------------------


def test_assign_io_ranks_follows_the_output_configuration() -> None:
    # two nodes, the first with one rank more
    node_ids = [0] * 5 + [1] * 4
    per_node = io_ranks.assign_io_ranks(common_io.IORanksPerNode(count=1), node_ids)
    assert per_node == io_ranks.assign_io_ranks_per_node(node_ids, 1)
    assert per_node.io_ranks == (4, 8)

    # a total number of IO ranks: on the fullest nodes
    in_total = io_ranks.assign_io_ranks(common_io.IORanksInTotal(count=1), node_ids)
    assert in_total == io_ranks.assign_io_ranks_on_fullest_nodes(node_ids, 1)
    assert in_total.io_ranks == (4,)


# --------------------------------------------------------------------------------------
# the layout of the roles on the nodes
# --------------------------------------------------------------------------------------


def tasks_per_node(*num_tasks: int) -> list[int]:
    """Block placement of the given numbers of tasks on consecutive nodes."""
    return [node for node, count in enumerate(num_tasks) for _ in range(count)]


@pytest.mark.parametrize(
    "assignment, num_ranks_per_node, layout",
    [
        (
            io_ranks.assign_io_ranks_per_node(block_layout(5, 10), 1),
            [(4, 1)] * 10,
            "40 compute rank(s) and 10 IO rank(s) on 10 node(s); compute+IO ranks per node: "
            "4+1 (x10)",
        ),
        (
            io_ranks.assign_io_ranks_on_fullest_nodes(tasks_per_node(5, 5, *[4] * 8), 2),
            [(4, 1)] * 2 + [(4, 0)] * 8,
            "40 compute rank(s) and 2 IO rank(s) on 10 node(s); compute+IO ranks per node: "
            "4+1 (x2), 4+0 (x8)",
        ),
        (
            io_ranks.assign_io_ranks_on_fullest_nodes(tasks_per_node(5, 5, *[4] * 8), 3),
            [(3, 2), (4, 1)] + [(4, 0)] * 8,
            "39 compute rank(s) and 3 IO rank(s) on 10 node(s); compute+IO ranks per node: "
            "3+2, 4+1, 4+0 (x8)",
        ),
        (
            io_ranks.assign_io_ranks_per_node(cyclic_layout(5, 2), 1),
            [(4, 1)] * 2,
            "8 compute rank(s) and 2 IO rank(s) on 2 node(s); compute+IO ranks per node: 4+1 (x2)",
        ),
    ],
    ids=["one-per-node", "surplus-tasks", "one-io-rank-too-many", "cyclic"],
)
def test_layout_counts_the_ranks_of_both_roles_per_node(
    assignment: io_ranks.IORankAssignment, num_ranks_per_node: list[tuple[int, int]], layout: str
) -> None:
    assert io_ranks.num_ranks_per_node(assignment) == num_ranks_per_node
    assert io_ranks.describe_layout(assignment) == layout


def test_log_layout_is_silent_about_evenly_filled_nodes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assignment = io_ranks.assign_io_ranks_on_fullest_nodes(tasks_per_node(5, 5, 4, 4), 2)
    with caplog.at_level(logging.INFO, logger=io_ranks.__name__):
        io_ranks.log_layout(assignment)
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.INFO, f"Dedicated IO ranks: {io_ranks.describe_layout(assignment)}.")
    ]


@pytest.mark.parametrize(
    "assignment, per_node",
    [
        # one IO rank more than surplus tasks: it takes a compute rank of the first node
        (
            io_ranks.assign_io_ranks_on_fullest_nodes(tasks_per_node(5, 5, 4, 4), 3),
            "3+2, 4+1, 4+0 (x2)",
        ),
        # IO ranks on every node of a launch with surplus tasks on two nodes only
        (io_ranks.assign_io_ranks_per_node(tasks_per_node(5, 5, 4, 4), 1), "4+1 (x2), 3+1 (x2)"),
        # the surplus tasks on nodes of their own are not the IO ranks
        (
            io_ranks.assign_io_ranks_on_fullest_nodes(tasks_per_node(4, 4, 4, 1), 1),
            "3+1, 4+0 (x2), 1+0",
        ),
    ],
    ids=["one-io-rank-too-many", "per-node-on-uneven-launch", "single-task-node"],
)
def test_log_layout_warns_about_unequal_numbers_of_compute_ranks(
    assignment: io_ranks.IORankAssignment,
    per_node: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the warning carries the layout itself: it is still there when INFO is filtered out
    with caplog.at_level(logging.WARNING, logger=io_ranks.__name__):
        io_ranks.log_layout(assignment)
    (warning,) = caplog.records
    assert warning.levelno == logging.WARNING
    assert "unequal numbers of compute ranks" in warning.getMessage()
    assert f"compute+IO ranks per node: {per_node})" in warning.getMessage()


@pytest.mark.parametrize(
    "assignment, per_node",
    [
        # no task for the IO rank: it takes a compute rank on every node
        (io_ranks.assign_io_ranks_per_node(tasks_per_node(4, 4, 4, 4), 1), "3+1 (x4)"),
        (io_ranks.assign_io_ranks_on_fullest_nodes(tasks_per_node(4, 4, 4, 4), 4), "3+1 (x4)"),
    ],
    ids=["per-node", "total"],
)
def test_log_layout_cannot_warn_about_a_launch_short_of_tasks_on_every_node(
    assignment: io_ranks.IORankAssignment,
    per_node: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The nodes hold equal numbers of compute ranks: only the layout line shows the launch."""
    with caplog.at_level(logging.INFO, logger=io_ranks.__name__):
        io_ranks.log_layout(assignment)
    (info,) = caplog.records
    assert info.levelno == logging.INFO
    assert info.getMessage().endswith(f"compute+IO ranks per node: {per_node}.")


# --------------------------------------------------------------------------------------
# the roles of the ranks of a run
# --------------------------------------------------------------------------------------


def output_config(io_ranks_config: common_io.IORanks | None) -> common_io.IOConfig:
    return common_io.IOConfig(
        field_groups=[common_io.FieldGroupIOConfig(basename="output", variables=["air_density"])],
        io_ranks=io_ranks_config,
    )


@pytest.mark.parametrize("io_config", [None, output_config(None)], ids=["no-output", "no-io-ranks"])
def test_rank_roles_without_io_ranks_leave_the_communicator_alone(
    io_config: common_io.IOConfig | None,
) -> None:
    process_props = decomp_defs.SingleNodeProcessProperties()
    rank_roles = io_ranks.create_rank_roles(io_config, process_props)
    assert isinstance(rank_roles, io_ranks.NoIORanks)
    assert not rank_roles.is_io_rank
    # every rank runs the model on the run's communicator
    assert rank_roles.process_props is process_props


def test_rank_roles_with_io_ranks_split_the_communicator_as_configured() -> None:
    process_props = decomp_defs.SingleNodeProcessProperties()
    node_ids = tasks_per_node(5, 4)
    with (
        mock.patch.object(io_ranks, "allgather_node_ids", return_value=node_ids),
        mock.patch.object(io_ranks, "split_io_ranks") as mock_split,
    ):
        rank_roles = io_ranks.create_rank_roles(
            output_config(common_io.IORanksInTotal(count=1)), process_props
        )
    assert rank_roles is mock_split.return_value
    mock_split.assert_called_once_with(
        process_props, io_ranks.assign_io_ranks_on_fullest_nodes(node_ids, 1)
    )


def test_no_io_ranks_let_a_failure_propagate_and_report_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    process_props = mock.MagicMock(rank=0, comm_size=4)
    rank_roles = io_ranks.NoIORanks(process_props)
    failure = RuntimeError("the model failed")
    with pytest.raises(RuntimeError) as raised, rank_roles.abort_on_failure():
        raise failure
    assert raised.value is failure
    # every rank raises the failure itself: nobody aborts the others
    process_props.comm.Abort.assert_not_called()
    with caplog.at_level(logging.INFO, logger=io_ranks.__name__):
        rank_roles.report_layout()
    assert not caplog.records


@pytest.fixture
def alarms(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The seconds of every ``signal.alarm`` call of the test, recorded in place of armed.

    An abort through a stand-in communicator returns, and its alarm would end pytest.
    """
    seconds: list[int] = []
    monkeypatch.setattr(signal, "alarm", seconds.append)
    return seconds


@pytest.mark.parametrize("abort_raises", [False, True], ids=["abort-returns", "abort-raises"])
def test_abort_arms_the_alarm_before_the_abort_and_never_cancels_it(
    abort_raises: bool, alarms: list[int]
) -> None:
    """A rank whose ``Abort`` blocks, returns or raises is still ended by the alarm."""
    alarms_at_abort: list[int] = []

    def abort(errorcode: int) -> None:
        alarms_at_abort.extend(alarms)
        if abort_raises:
            raise RuntimeError("the abort failed")

    comm = mock.MagicMock()
    comm.Abort.side_effect = abort
    if abort_raises:
        with pytest.raises(RuntimeError, match="the abort failed"):
            io_ranks._abort(comm)
    else:
        io_ranks._abort(comm)
    comm.Abort.assert_called_once_with(1)
    assert alarms_at_abort == [io_ranks._ABORT_GRACE_SECONDS]
    assert alarms == [io_ranks._ABORT_GRACE_SECONDS]


def io_rank_split(
    *, is_io_rank: bool, role_rank: int, run_props: mock.MagicMock
) -> io_ranks.IORankSplit:
    """A split of a 5+4 run whose communicators are stand-ins."""
    return io_ranks.IORankSplit(
        run_props=run_props,
        assignment=io_ranks.assign_io_ranks_per_node(tasks_per_node(5, 4), 1),
        is_io_rank=is_io_rank,
        process_props=mock.MagicMock(rank=role_rank),
        intercomm=mock.MagicMock(),
        io_rank=None if is_io_rank else 0,
        served_compute_ranks=(0, 1, 2, 3) if is_io_rank else (),
    )


@pytest.mark.parametrize("is_io_rank", [False, True], ids=["compute-rank", "io-rank"])
def test_io_rank_split_aborts_the_whole_run_on_failure(is_io_rank: bool, alarms: list[int]) -> None:
    run_props = mock.MagicMock()
    rank_roles = io_rank_split(is_io_rank=is_io_rank, role_rank=1, run_props=run_props)
    failure = RuntimeError("the rank failed")
    with pytest.raises(RuntimeError) as raised, rank_roles.abort_on_failure():
        raise failure
    assert raised.value is failure
    # neither role can observe the other's failure: all ranks of both roles go down
    run_props.comm.Abort.assert_called_once_with(1)
    rank_roles.process_props.comm.Abort.assert_not_called()
    # an abort that blocks in an exit handler must not keep the rank alive (``_abort``)
    assert alarms == [io_ranks._ABORT_GRACE_SECONDS]

    with rank_roles.abort_on_failure():
        pass
    run_props.comm.Abort.assert_called_once()
    assert len(alarms) == 1


@pytest.mark.parametrize(
    "is_io_rank, role_rank, reports",
    [(False, 0, True), (False, 1, False), (True, 0, False)],
    ids=["compute-rank-0", "other-compute-rank", "io-rank-0"],
)
def test_io_rank_split_reports_the_layout_on_one_rank(
    is_io_rank: bool, role_rank: int, reports: bool
) -> None:
    rank_roles = io_rank_split(
        is_io_rank=is_io_rank, role_rank=role_rank, run_props=mock.MagicMock()
    )
    with mock.patch.object(io_ranks, "log_layout") as mock_log_layout:
        rank_roles.report_layout()
    if reports:
        mock_log_layout.assert_called_once_with(rank_roles.assignment)
    else:
        mock_log_layout.assert_not_called()
