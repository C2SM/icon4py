# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the split of a communicator into compute ranks and dedicated IO ranks (``io.io_ranks``)."""

import functools
from collections.abc import Callable

import numpy as np
import pytest

import icon4py.model.common.exceptions as errors
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.io import io_ranks

from ...fixtures import process_props


@pytest.mark.mpi(min_size=2)
@pytest.mark.level("unit")
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "assign, pairs_of_ranks_as_nodes",
    [
        (functools.partial(io_ranks.assign_io_ranks_per_node, io_ranks_per_node=1), False),
        (functools.partial(io_ranks.assign_io_ranks_per_node, io_ranks_per_node=2), False),
        (
            functools.partial(io_ranks.assign_io_ranks_on_fullest_nodes, num_io_ranks=1),
            False,
        ),
        # with every pair of ranks posing as a node the IO ranks are not the last ranks
        # of the run, and compute ranks ship to an IO rank of another "node"
        (functools.partial(io_ranks.assign_io_ranks_per_node, io_ranks_per_node=1), True),
        (
            functools.partial(io_ranks.assign_io_ranks_on_fullest_nodes, num_io_ranks=1),
            True,
        ),
    ],
    ids=[
        "one-per-node",
        "two-per-node",
        "one-on-the-fullest-node",
        "one-per-pair",
        "one-on-a-pair",
    ],
)
def test_split_io_ranks_roles_and_intercomm(
    process_props: decomp_defs.ProcessProperties,
    assign: Callable[[list[int]], io_ranks.IORankAssignment],
    pairs_of_ranks_as_nodes: bool,
) -> None:
    node_ids = io_ranks.allgather_node_ids(process_props)
    assert len(node_ids) == process_props.comm_size
    # a node is identified by its lowest rank
    my_node = node_ids[process_props.rank]
    assert my_node == min(rank for rank, node in enumerate(node_ids) if node == my_node)
    if pairs_of_ranks_as_nodes:
        node_ids = [rank // 2 for rank in range(process_props.comm_size)]
    try:
        assignment = assign(node_ids)
    except errors.InvalidConfigError as error:
        pytest.skip(f"too few ranks for these IO ranks: {error}")
    split = io_ranks.split_io_ranks(process_props, assignment)

    # the roles follow the assignment; each role is one communicator keeping the rank order
    assert split.run_props is process_props
    assert split.assignment is assignment
    assert split.is_io_rank == (process_props.rank in assignment.io_ranks)
    role_ranks = assignment.io_ranks if split.is_io_rank else assignment.compute_ranks
    expected_name = io_ranks.IO_COMM_NAME if split.is_io_rank else io_ranks.COMPUTE_COMM_NAME
    assert split.process_props.comm_size == len(role_ranks)
    assert split.process_props.comm_name == expected_name
    assert split.process_props.rank == role_ranks.index(process_props.rank)

    # who serves whom
    if split.is_io_rank:
        assert split.io_rank is None
        assert split.served_compute_ranks == assignment.served_compute_ranks(
            split.process_props.rank
        )
        assert split.served_compute_ranks
    else:
        assert split.served_compute_ranks == ()
        assert split.io_rank == assignment.io_rank_of(split.process_props.rank)

    # the inter-communicator connects the two roles
    assert split.intercomm.Is_inter()
    assert split.intercomm.Get_rank() == split.process_props.rank
    assert split.intercomm.Get_size() == len(role_ranks)
    assert split.intercomm.Get_remote_size() == process_props.comm_size - len(role_ranks)
    # ...and carries messages between them: every compute rank tells its rank to its IO rank
    # (buffer messages, as the output protocol uses: mpi4py's object receive goes
    # through a matched probe, which hangs under Open MPI 5 with libfabric's CXI provider)
    if split.is_io_rank:
        received = []
        for rank in split.served_compute_ranks:
            message = np.empty(1, dtype=np.int64)
            split.intercomm.Recv(message, source=rank, tag=7)
            received.append(int(message[0]))
        assert received == list(split.served_compute_ranks)
    else:
        message = np.array([split.process_props.rank], dtype=np.int64)
        split.intercomm.Send(message, dest=split.io_rank, tag=7)
    process_props.comm.Barrier()
