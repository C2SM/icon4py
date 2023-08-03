# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from atm_dyn_iconam.tests.mpi_tests.common import (  # noqa F401
    data_path,
    download_data,
    props,
)
from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.decomposition.decomposed import (
    DecompositionInfo,
    DomainDescriptorIdGenerator,
    GHexMultiNode,
    SingleNode,
    create_exchange,
)
from icon4py.decomposition.parallel_setup import ProcessProperties
from icon4py.driver.io_utils import (
    SerializationType,
    read_decomp_info,
    read_icon_grid,
)


"""
running tests with mpi:

mpirun -np 2 python -m pytest -v --with-mpi tests/mpi_tests/test_parallel_setup.py

mpirun -np 2 pytest -v --with-mpi tests/mpi_tests/


"""


@pytest.mark.mpi
@pytest.mark.skipif(props.comm_size != 2, reason="runs on 2 nodes only")
@pytest.mark.parametrize(
    ("dim, owned, total"),
    (
        (CellDim, (10448, 10448), (10611, 10612)),
        (EdgeDim, (15820, 15738), (16065, 16067)),
        (VertexDim, (5373, 5290), (5455, 5456)),
    ),
)
def test_decomposition_info_masked(
    dim, owned, total, caplog, download_data  # noqa F811
):
    my_rank = props.rank
    decomposition_info = read_decomp_info(data_path, props, SerializationType.SB)
    all_indices = decomposition_info.global_index(dim, DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]
    assert all_indices.shape[0] == my_total

    owned_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == my_owned

    halo_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.HALO
    )
    assert halo_indices.shape[0] == my_total - my_owned
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)


@pytest.mark.skipif(props.comm_size != 2, reason="runs on 2 nodes only")
@pytest.mark.parametrize(
    ("dim, owned, total"),
    (
        (CellDim, (10448, 10448), (10611, 10612)),
        (EdgeDim, (15820, 15738), (16065, 16067)),
        (VertexDim, (5373, 5290), (5455, 5456)),
    ),
)
def test_decomposition_info_local_index(
    dim, owned, total, caplog, download_data  # noqa F811
):
    my_rank = props.rank
    decomposition_info = read_decomp_info(data_path, props, SerializationType.SB)
    all_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]

    assert all_indices.shape[0] == my_total
    assert np.array_equal(all_indices, np.arange(0, my_total))
    halo_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.HALO)
    assert halo_indices.shape[0] == my_total - my_owned
    assert halo_indices.shape[0] < all_indices.shape[0]
    assert np.alltrue(halo_indices <= np.max(all_indices))

    owned_indices = decomposition_info.local_index(
        dim, DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == my_owned
    assert owned_indices.shape[0] <= all_indices.shape[0]
    assert np.alltrue(owned_indices <= np.max(all_indices))
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)


def _assert_index_partitioning(all_indices, halo_indices, owned_indices):
    owned_list = owned_indices.tolist()
    halos_list = halo_indices.tolist()
    all_list = all_indices.tolist()
    assert set(owned_list) & set(halos_list) == set()
    assert set(owned_list) & set(all_list) == set(owned_list)
    assert set(halos_list) & set(all_list) == set(halos_list)
    assert set(halos_list) | set(owned_list) == set(all_list)


@pytest.mark.mpi
@pytest.mark.parametrize("num", [1, 2, 3])
def test_domain_descriptor_id_are_globally_unique(num):

    size = props.comm_size
    id_gen = DomainDescriptorIdGenerator(parallel_props=props)
    id1 = id_gen()
    assert id1 == props.comm_size * props.rank
    assert id1 < props.comm_size * (props.rank + 1)
    ids = []
    ids.append(id1)
    for _ in range(1, num * size):
        next_id = id_gen()
        assert next_id > id1
        ids.append(next_id)
    all_ids = props.comm.gather(ids, root=0)
    if props.rank == 0:
        all_ids = np.asarray(all_ids).flatten()
        assert len(all_ids) == size * size * num
        assert len(all_ids) == len(set(all_ids))


@pytest.mark.mpi
@pytest.mark.skipif(
    props.comm_size not in (1, 2, 4),
    reason="input files only available for 1 or 2 nodes",
)
def test_decomposition_info_matches_gridsize(caplog, download_data):  # noqa F811
    decomposition_info = read_decomp_info(
        data_path,
        props,
        SerializationType.SB,
    )
    icon_grid = read_icon_grid(data_path, props.rank)
    assert (
        decomposition_info.global_index(
            dim=CellDim, entry_type=DecompositionInfo.EntryType.ALL
        ).shape[0]
        == icon_grid.num_cells()
    )
    assert (
        decomposition_info.global_index(
            VertexDim, DecompositionInfo.EntryType.ALL
        ).shape[0]
        == icon_grid.num_vertices()
    )
    assert (
        decomposition_info.global_index(EdgeDim, DecompositionInfo.EntryType.ALL).shape[
            0
        ]
        == icon_grid.num_edges()
    )


@pytest.mark.mpi
def test_create_multi_pytenode_runtime_with_mpi(download_data):  # noqa F811
    decomp_info = read_decomp_info(data_path, props)
    exchange = create_exchange(props, decomp_info)
    if props.comm_size > 1:
        assert isinstance(exchange, GHexMultiNode)
    else:
        assert isinstance(exchange, SingleNode)


def test_create_single_node_runtime_without_mpi():
    props = ProcessProperties.from_single_node()
    decomp_info = read_decomp_info(data_path, props)
    exchange = create_exchange(props, decomp_info)

    assert isinstance(exchange, SingleNode)
