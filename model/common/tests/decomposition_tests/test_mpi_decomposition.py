# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common.utils.data_allocation import constant_field


try:
    import mpi4py  # noqa: F401 # import mpi4py to check for optional mpi dependency
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition.definitions import (
    DecompositionInfo,
    DomainDescriptorIdGenerator,
    SingleNodeExchange,
    create_exchange,
)
from icon4py.model.common.decomposition.mpi_decomposition import GHexMultiNodeExchange
from icon4py.model.testing.datatest_fixtures import (  # noqa: F401 # import fixtures from test_utils
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    ranked_data_path,
)
from icon4py.model.testing.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
)


"""
running tests with mpi:

mpirun -np 2 python -m pytest -v --with-mpi tests/mpi_tests/test_parallel_setup.py

mpirun -np 2 pytest -v --with-mpi tests/mpi_tests/


"""


@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_props(processor_props):  # noqa: F811  # fixture
    assert processor_props.comm


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    ("dim, owned, total"),
    (
        (dims.CellDim, (10448, 10448), (10611, 10612)),
        (dims.EdgeDim, (15820, 15738), (16065, 16067)),
        (dims.VertexDim, (5373, 5290), (5455, 5456)),
    ),
)
@pytest.mark.datatest
def test_decomposition_info_masked(
    dim,
    owned,
    total,
    caplog,
    download_ser_data,  # noqa: F811 # fixture
    decomposition_info,  # noqa: F811 # fixture
    processor_props,  # noqa: F811 # fixture
):
    check_comm_size(processor_props, sizes=[2])
    my_rank = processor_props.rank
    all_indices = decomposition_info.global_index(dim, DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]
    assert all_indices.shape[0] == my_total

    owned_indices = decomposition_info.global_index(dim, DecompositionInfo.EntryType.OWNED)
    assert owned_indices.shape[0] == my_owned

    halo_indices = decomposition_info.global_index(dim, DecompositionInfo.EntryType.HALO)
    assert halo_indices.shape[0] == my_total - my_owned
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)


def _assert_index_partitioning(all_indices, halo_indices, owned_indices):
    owned_list = owned_indices.tolist()
    halos_list = halo_indices.tolist()
    all_list = all_indices.tolist()
    assert set(owned_list) & set(halos_list) == set()
    assert set(owned_list) & set(all_list) == set(owned_list)
    assert set(halos_list) & set(all_list) == set(halos_list)
    assert set(halos_list) | set(owned_list) == set(all_list)


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    ("dim, owned, total"),
    (
        (dims.CellDim, (10448, 10448), (10611, 10612)),
        (dims.EdgeDim, (15820, 15738), (16065, 16067)),
        (dims.VertexDim, (5373, 5290), (5455, 5456)),
    ),
)
@pytest.mark.datatest
@pytest.mark.mpi(min_size=2)
def test_decomposition_info_local_index(
    dim,
    owned,
    total,
    caplog,
    download_ser_data,  # noqa: F811 #fixture
    decomposition_info,  # noqa: F811 #fixture
    processor_props,  # noqa: F811 #fixture
):
    check_comm_size(processor_props, sizes=[2])
    my_rank = processor_props.rank
    all_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]

    assert all_indices.shape[0] == my_total
    assert np.array_equal(all_indices, np.arange(0, my_total))
    halo_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.HALO)
    assert halo_indices.shape[0] == my_total - my_owned
    assert halo_indices.shape[0] < all_indices.shape[0]
    assert np.all(halo_indices <= np.max(all_indices))

    owned_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.OWNED)
    assert owned_indices.shape[0] == my_owned
    assert owned_indices.shape[0] <= all_indices.shape[0]
    assert np.all(owned_indices <= np.max(all_indices))
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("num", [1, 2, 3, 4, 5, 6, 7, 8])
def test_domain_descriptor_id_are_globally_unique(
    num,
    processor_props,  # noqa F811 #fixture
):
    props = processor_props
    size = props.comm_size
    id_gen = DomainDescriptorIdGenerator(parallel_props=props)
    id1 = id_gen()
    assert id1 == props.comm_size * props.rank
    assert id1 < props.comm_size * (props.rank + 2)
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
@pytest.mark.datatest
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_decomposition_info_matches_gridsize(
    caplog,
    download_ser_data,  # noqa: F811 #fixture
    decomposition_info,  # noqa: F811 #fixture
    icon_grid,  # noqa: F811 #fixture
    processor_props,  # noqa: F811 #fixture
):
    check_comm_size(processor_props)
    assert (
        decomposition_info.global_index(
            dim=dims.CellDim, entry_type=DecompositionInfo.EntryType.ALL
        ).shape[0]
        == icon_grid.num_cells
    )
    assert (
        decomposition_info.global_index(dims.VertexDim, DecompositionInfo.EntryType.ALL).shape[0]
        == icon_grid.num_vertices
    )
    assert (
        decomposition_info.global_index(dims.EdgeDim, DecompositionInfo.EntryType.ALL).shape[0]
        == icon_grid.num_edges
    )


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_create_multi_node_runtime_with_mpi(
    decomposition_info,  # noqa: F811 # fixture
    processor_props,  # noqa: F811  # fixture
):
    props = processor_props
    exchange = create_exchange(props, decomposition_info)
    if props.comm_size > 1:
        assert isinstance(exchange, GHexMultiNodeExchange)
    else:
        assert isinstance(exchange, SingleNodeExchange)


@pytest.mark.parametrize("processor_props", [False], indirect=True)
@pytest.mark.mpi_skip()
def test_create_single_node_runtime_without_mpi(
    processor_props,  # noqa: F811 # fixture
    decomposition_info,  # noqa: F811 # fixture
):
    exchange = create_exchange(processor_props, decomposition_info)
    assert isinstance(exchange, SingleNodeExchange)


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("dimension", (dims.CellDim, dims.VertexDim, dims.EdgeDim))
def test_exchange_on_dummy_data(
    processor_props,  # noqa: F811 # fixture
    decomposition_info,  # noqa: F811 # fixture
    grid_savepoint,  # noqa: F811 # fixture
    metrics_savepoint,  # noqa: F811 # fixture
    dimension,
):
    exchange = create_exchange(processor_props, decomposition_info)
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)

    number = processor_props.rank + 10.0
    input_field = constant_field(
        grid,
        number,
        dimension,
        dims.KDim,
    )

    halo_points = decomposition_info.local_index(dimension, DecompositionInfo.EntryType.HALO)
    local_points = decomposition_info.local_index(dimension, DecompositionInfo.EntryType.OWNED)
    assert np.all(input_field == number)
    exchange.exchange_and_wait(dimension, input_field)
    result = input_field.asnumpy()
    print(f"rank={processor_props.rank} - num of halo points ={halo_points.shape}")
    print(
        f" rank={processor_props.rank} - exchanged points: {np.sum(result != number)/grid.num_levels}"
    )
    print(f"rank={processor_props.rank} - halo points: {halo_points}")

    assert np.all(result[local_points, :] == number)
    assert np.all(result[halo_points, :] != number)

    changed_points = np.argwhere(result[:, 2] != number)
    print(f"rank={processor_props.rank} - num changed points {changed_points.shape} ")

    print(f"rank={processor_props.rank} - changed points {changed_points} ")
