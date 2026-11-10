# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""MPI-free tests of the choice of dedicated IO ranks (``mpi_decomposition``)."""

import collections
from collections.abc import Callable

import pytest

import icon4py.model.common.exceptions as errors
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition


def block_layout(ranks_per_node: int, num_nodes: int) -> list[int]:
    """Consecutive ranks share a node: the launchers' default placement."""
    return [rank // ranks_per_node for rank in range(ranks_per_node * num_nodes)]


def cyclic_layout(ranks_per_node: int, num_nodes: int) -> list[int]:
    """Consecutive ranks go to different nodes (``--distribution=cyclic``)."""
    return [rank % num_nodes for rank in range(ranks_per_node * num_nodes)]


def served_ranks(assignment: mpi_decomposition.IORankAssignment) -> dict[int, list[int]]:
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
    assignment = mpi_decomposition.assign_io_ranks_per_node(node_ids, io_ranks_per_node)
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
    assignment = mpi_decomposition.assign_io_ranks_per_node([0] * 6, 2)
    assert assignment.io_ranks == (4, 5)
    assert assignment.compute_ranks == (0, 1, 2, 3)
    assert [assignment.io_rank_of(compute_rank) for compute_rank in range(4)] == [0, 0, 1, 1]
    assert assignment.served_compute_ranks(0) == (0, 1)
    assert assignment.served_compute_ranks(1) == (2, 3)


def test_assign_io_ranks_rejects_nodes_short_of_compute_ranks() -> None:
    # the node of rank 4 has two ranks: two IO ranks would leave it no compute rank
    with pytest.raises(errors.InvalidConfigError, match="node of rank 4 has 2 ranks"):
        mpi_decomposition.assign_io_ranks_per_node([0, 0, 0, 0, 1, 1], 2)
    with pytest.raises(errors.InvalidConfigError, match="at least 1"):
        mpi_decomposition.assign_io_ranks_per_node([0, 0], 0)


# --------------------------------------------------------------------------------------
# A total number of IO ranks, on the fullest nodes
# --------------------------------------------------------------------------------------


def test_fullest_nodes_turn_the_surplus_tasks_into_io_ranks() -> None:
    # ten nodes with four compute ranks each and two tasks more, which the launcher
    # put on the first two nodes
    node_ids = [0] * 5 + [1] * 5 + [node for node in range(2, 10) for _ in range(4)]
    assignment = mpi_decomposition.assign_io_ranks_on_fullest_nodes(node_ids, 2)

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
    assignment = mpi_decomposition.assign_io_ranks_on_fullest_nodes(node_ids, 2)
    assert assignment.io_ranks == (8, 9)
    assert served_ranks(assignment) == {8: [0, 1, 4, 5], 9: [2, 3, 6, 7]}


def test_fullest_nodes_spread_the_io_ranks_over_equally_full_nodes() -> None:
    # no node holds more ranks than the others: the nodes 0 and 5 give their last rank
    assignment = mpi_decomposition.assign_io_ranks_on_fullest_nodes(block_layout(4, 10), 2)
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
    assert mpi_decomposition.assign_io_ranks_on_fullest_nodes(
        node_ids, num_io_ranks
    ) == mpi_decomposition.assign_io_ranks_per_node(node_ids, io_ranks_per_node)


# --------------------------------------------------------------------------------------
# A total number of IO ranks, the last ranks of the run
# --------------------------------------------------------------------------------------


def test_last_ranks_on_a_node_of_their_own_share_the_compute_ranks_in_blocks() -> None:
    # ten nodes with four compute ranks each, then two IO tasks on an eleventh node
    node_ids = [*block_layout(4, 10), 10, 10]
    assignment = mpi_decomposition.assign_last_ranks_as_io_ranks(node_ids, 2)
    assert assignment.io_ranks == (40, 41)
    assert assignment.compute_ranks == tuple(range(40))
    assert served_ranks(assignment) == {40: list(range(20)), 41: list(range(20, 40))}


def test_last_ranks_placed_on_compute_nodes_serve_their_node_first() -> None:
    # the launcher put the two IO tasks on the last and on the first compute node
    node_ids = [*block_layout(4, 10), 9, 0]
    assignment = mpi_decomposition.assign_last_ranks_as_io_ranks(node_ids, 2)
    assert assignment.io_ranks == (40, 41)
    assert served_ranks(assignment) == {
        40: [*range(4, 20), 36, 37, 38, 39],
        41: [0, 1, 2, 3, *range(20, 36)],
    }


def test_last_ranks_do_not_depend_on_the_placement() -> None:
    assignment = mpi_decomposition.assign_last_ranks_as_io_ranks(cyclic_layout(3, 2), 2)
    assert assignment.io_ranks == (4, 5)
    # each IO rank serves the compute ranks of its node
    assert served_ranks(assignment) == {4: [0, 2], 5: [1, 3]}


# --------------------------------------------------------------------------------------
# All choices
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "assign",
    [
        mpi_decomposition.assign_io_ranks_per_node,
        mpi_decomposition.assign_io_ranks_on_fullest_nodes,
        mpi_decomposition.assign_last_ranks_as_io_ranks,
    ],
)
@pytest.mark.parametrize(
    "node_ids",
    [block_layout(6, 3), cyclic_layout(6, 3), [0] * 7 + [1] * 6 + [2] * 6 + [3] * 7],
    ids=["block", "cyclic", "uneven"],
)
def test_every_choice_partitions_the_ranks_and_serves_all_compute_ranks(
    assign: Callable[[list[int], int], mpi_decomposition.IORankAssignment], node_ids: list[int]
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


@pytest.mark.parametrize(
    "assign",
    [
        mpi_decomposition.assign_io_ranks_on_fullest_nodes,
        mpi_decomposition.assign_last_ranks_as_io_ranks,
    ],
)
def test_total_io_ranks_reject_runs_short_of_compute_ranks(
    assign: Callable[[list[int], int], mpi_decomposition.IORankAssignment],
) -> None:
    with pytest.raises(errors.InvalidConfigError, match="at least 1"):
        assign([0, 0], 0)
    # three IO ranks need three compute ranks
    with pytest.raises(errors.InvalidConfigError, match="run has 5 ranks"):
        assign([0, 0, 0, 1, 1], 3)


def test_io_rank_without_a_compute_rank_is_rejected() -> None:
    # the last two ranks are IO ranks on a node without compute ranks, and the compute
    # ranks of the first node ship to its own IO rank
    with pytest.raises(errors.InvalidConfigError, match="Rank 5 would be an IO rank without"):
        mpi_decomposition.assign_last_ranks_as_io_ranks([0, 0, 0, 0, 0, 1, 1], 3)


def test_node_ids_of_a_single_rank_run_need_no_mpi() -> None:
    assert mpi_decomposition.allgather_node_ids(decomp_defs.SingleNodeProcessProperties()) == [0]


def test_split_io_ranks_requires_an_mpi_run() -> None:
    assignment = mpi_decomposition.assign_io_ranks_per_node([0, 0], 1)
    with pytest.raises(errors.InvalidConfigError, match="require an MPI run"):
        mpi_decomposition.split_io_ranks(decomp_defs.SingleNodeProcessProperties(), assignment)
