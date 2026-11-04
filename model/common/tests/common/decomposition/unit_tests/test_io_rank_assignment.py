# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""MPI-free tests of the node-aware choice of dedicated IO ranks (``mpi_decomposition``)."""

import collections

import pytest

import icon4py.model.common.exceptions as errors
from icon4py.model.common.decomposition import mpi_decomposition


def block_layout(ranks_per_node: int, num_nodes: int) -> list[int]:
    """Consecutive ranks share a node: the launchers' default placement."""
    return [rank // ranks_per_node for rank in range(ranks_per_node * num_nodes)]


def cyclic_layout(ranks_per_node: int, num_nodes: int) -> list[int]:
    """Consecutive ranks go to different nodes (``--distribution=cyclic``)."""
    return [rank % num_nodes for rank in range(ranks_per_node * num_nodes)]


@pytest.mark.parametrize(
    "node_ids",
    [block_layout(5, 2), cyclic_layout(5, 2), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]],
    ids=["block", "cyclic", "uneven"],
)
@pytest.mark.parametrize("io_ranks_per_node", [1, 2])
def test_assign_io_ranks_puts_io_ranks_on_every_node(
    node_ids: list[int], io_ranks_per_node: int
) -> None:
    assignment = mpi_decomposition.assign_io_ranks(node_ids, io_ranks_per_node)
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
    assignment = mpi_decomposition.assign_io_ranks([0] * 6, 2)
    assert assignment.io_ranks == (4, 5)
    assert assignment.compute_ranks == (0, 1, 2, 3)
    assert [assignment.io_rank_of(compute_rank) for compute_rank in range(4)] == [0, 0, 1, 1]
    assert assignment.served_compute_ranks(0) == (0, 1)
    assert assignment.served_compute_ranks(1) == (2, 3)


def test_assign_io_ranks_rejects_nodes_short_of_compute_ranks() -> None:
    # the node of rank 4 has two ranks: two IO ranks would leave it no compute rank
    with pytest.raises(errors.InvalidConfigError, match="node of rank 4 has 2 ranks"):
        mpi_decomposition.assign_io_ranks([0, 0, 0, 0, 1, 1], 2)
    with pytest.raises(errors.InvalidConfigError, match="at least 1"):
        mpi_decomposition.assign_io_ranks([0, 0], 0)
