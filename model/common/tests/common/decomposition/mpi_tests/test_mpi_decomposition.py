# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

import icon4py.model.testing.test_utils as test_helpers
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon
from icon4py.model.common.interpolation import interpolation_fields
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, parallel_helpers, serialbox

from ...fixtures import (
    backend,
    backend_like,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


_log = logging.getLogger(__name__)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_props(process_props: definitions.ProcessProperties) -> None:
    assert process_props.comm
    assert process_props.comm_size > 1
    assert 0 <= process_props.rank < process_props.comm_size


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "experiment",
    [
        test_defs.Experiments.MCH_CH_R04B09,
    ],
)
@pytest.mark.parametrize("process_props", [True], indirect=True)
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
    dim: gtx.Dimension,
    owned: int,
    total: int,
    caplog: Any,
    download_ser_data: Any,
    decomposition_info: definitions.DecompositionInfo,
    process_props: definitions.ProcessProperties,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(process_props, sizes=(2,))
    my_rank = process_props.rank
    all_indices = decomposition_info.global_index(dim, definitions.DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]
    assert all_indices.shape[0] == my_total

    owned_indices = decomposition_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == my_owned

    halo_indices = decomposition_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.HALO
    )
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


@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment",
    [
        test_defs.Experiments.MCH_CH_R04B09,
    ],
)
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
    dim: gtx.Dimension,
    owned: int,
    total: int,
    caplog: Any,
    decomposition_info: definitions.DecompositionInfo,
    process_props: definitions.ProcessProperties,
    experiment: test_defs.Experiment,
):
    caplog.set_level(logging.INFO)
    parallel_helpers.check_comm_size(process_props, sizes=(2,))
    my_rank = process_props.rank
    all_indices = decomposition_info.local_index(dim, definitions.DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]

    assert all_indices.shape[0] == my_total
    assert np.array_equal(all_indices, np.arange(0, my_total))
    halo_indices = decomposition_info.local_index(dim, definitions.DecompositionInfo.EntryType.HALO)
    assert halo_indices.shape[0] == my_total - my_owned
    assert halo_indices.shape[0] < all_indices.shape[0]
    assert np.all(halo_indices <= np.max(all_indices))

    owned_indices = decomposition_info.local_index(
        dim, definitions.DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == my_owned
    assert owned_indices.shape[0] <= all_indices.shape[0]
    assert np.all(owned_indices <= np.max(all_indices))
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("dim", (dims.CellDim, dims.EdgeDim, dims.VertexDim))
def test_decomposition_info_halo_level_mask(
    dim: gtx.Dimension,
    experiment: test_defs.Experiment,
    decomposition_info: definitions.DecompositionInfo,
) -> None:
    first_halo_level = decomposition_info.halo_level_mask(
        dim, definitions.DecompositionFlag.FIRST_HALO_LEVEL
    )
    assert first_halo_level.ndim == 1
    assert np.count_nonzero(first_halo_level) == decomposition_info.get_halo_size(
        dim, definitions.DecompositionFlag.FIRST_HALO_LEVEL
    )
    second_halo_level = decomposition_info.halo_level_mask(
        dim, definitions.DecompositionFlag.SECOND_HALO_LEVEL
    )
    assert second_halo_level.ndim == 1
    assert np.count_nonzero(second_halo_level) == decomposition_info.get_halo_size(
        dim, definitions.DecompositionFlag.SECOND_HALO_LEVEL
    )
    assert np.count_nonzero(first_halo_level) + np.count_nonzero(
        second_halo_level
    ) == np.count_nonzero(~decomposition_info.owner_mask(dim))


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("dim", (dims.CellDim, dims.EdgeDim, dims.VertexDim))
def test_decomposition_info_third_level_is_empty(
    dim: gtx.Dimension,
    experiment: test_defs.Experiment,
    decomposition_info: definitions.DecompositionInfo,
) -> None:
    level = decomposition_info.halo_level_mask(dim, definitions.DecompositionFlag.THIRD_HALO_LEVEL)
    assert np.count_nonzero(level) == 0


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("num", [1, 2, 3, 4, 5, 6, 7, 8])
def test_domain_descriptor_id_are_globally_unique(
    num: int,
    process_props: definitions.ProcessProperties,
) -> None:
    size = process_props.comm_size
    id_gen = definitions.DomainDescriptorIdGenerator(process_props=process_props)
    id1 = id_gen()
    assert id1 == process_props.comm_size * process_props.rank
    assert id1 < process_props.comm_size * (process_props.rank + 2)
    ids = []
    ids.append(id1)
    for _ in range(1, num * size):
        next_id = id_gen()
        assert next_id > id1
        ids.append(next_id)
    all_ids = process_props.comm.gather(ids, root=0)
    if process_props.rank == 0:
        all_ids = np.asarray(all_ids).flatten()
        assert len(all_ids) == size * size * num
        assert len(all_ids) == len(set(all_ids))


@pytest.mark.mpi
@pytest.mark.datatest
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_decomposition_info_matches_gridsize(
    caplog: Any,
    decomposition_info: definitions.DecompositionInfo,
    icon_grid: icon.IconGrid,
    process_props: definitions.ProcessProperties,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    assert (
        decomposition_info.global_index(
            dim=dims.CellDim, entry_type=definitions.DecompositionInfo.EntryType.ALL
        ).shape[0]
        == icon_grid.num_cells
    )
    assert (
        decomposition_info.global_index(
            dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL
        ).shape[0]
        == icon_grid.num_vertices
    )
    assert (
        decomposition_info.global_index(
            dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL
        ).shape[0]
        == icon_grid.num_edges
    )


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_create_multi_rank_runtime_with_mpi(
    decomposition_info: definitions.DecompositionInfo,
    process_props: definitions.ProcessProperties,
) -> None:
    exchange = definitions.create_exchange(process_props, decomposition_info)
    if process_props.comm_size > 1:
        assert isinstance(exchange, mpi_decomposition.GHexMultiNodeExchange)
    else:
        assert isinstance(exchange, definitions.SingleNodeExchange)


@pytest.mark.parametrize("process_props", [False], indirect=True)
@pytest.mark.mpi_skip()
def test_create_single_node_runtime_without_mpi(
    process_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,
) -> None:
    exchange = definitions.create_exchange(process_props, decomposition_info)
    assert isinstance(exchange, definitions.SingleNodeExchange)


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("dimension", (dims.CellDim, dims.EdgeDim, dims.VertexDim))
def test_exchange_on_dummy_data(
    process_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,
    grid_savepoint: serialbox.IconGridSavepoint,
    dimension: gtx.Dimension,
    backend: gtx.typing.Backend | None,
) -> None:
    exchange = definitions.create_exchange(process_props, decomposition_info)
    grid = grid_savepoint.construct_icon_grid(backend=backend)

    number = process_props.rank + 10
    input_field = data_alloc.constant_field(
        grid,
        number,
        dimension,
        dims.KDim,
        allocator=backend,
    )

    halo_points = data_alloc.as_numpy(
        decomposition_info.local_index(dimension, definitions.DecompositionInfo.EntryType.HALO)
    )
    local_points = data_alloc.as_numpy(
        decomposition_info.local_index(dimension, definitions.DecompositionInfo.EntryType.OWNED)
    )
    assert (input_field.ndarray == number).all()
    exchange.exchange(dimension, input_field, stream=definitions.BLOCK)
    result = input_field.asnumpy()
    _log.info(f"rank={process_props.rank} - num of halo points ={halo_points.shape}")
    _log.info(
        f" rank={process_props.rank} - exchanged points: {np.sum(result != number) / grid.num_levels}"
    )
    _log.info(f"rank={process_props.rank} - halo points: {halo_points}")
    changed_points = np.argwhere(result[:, 2] != number)
    _log.info(f"rank={process_props.rank} - num changed points {changed_points.shape} ")

    assert (result[local_points, :] == number).all()
    assert (result[halo_points, :] != number).all()


@pytest.mark.mpi
@pytest.mark.datatest
@pytest.mark.embedded_only
@pytest.mark.parametrize("process_props", [False], indirect=True)
def test_halo_exchange_for_sparse_field(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: test_defs.Experiment,
    process_props: definitions.ProcessProperties,
    grid_savepoint: serialbox.IconGridSavepoint,
    icon_grid: icon.IconGrid,
    decomposition_info: definitions.DecompositionInfo,
):
    edge_length = grid_savepoint.primal_edge_length()
    edge_orientation = grid_savepoint.edge_orientation()
    area = grid_savepoint.cell_areas()
    field_ref = interpolation_savepoint.geofac_div()
    _log.info(
        f"{process_props.rank}/{process_props.comm_size}: size of reference field {field_ref.asnumpy().shape}"
    )
    result = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.C2EDim, dtype=gtx.float64, allocator=None
    )
    exchange = definitions.create_exchange(process_props, decomposition_info)

    # mandatory computation on embedded because the result is sparse
    interpolation_fields.compute_geofac_div.with_backend(None)(
        edge_length,
        edge_orientation,
        area,
        out=result,
        offset_provider={"C2E": icon_grid.get_connectivity("C2E")},
    )
    _log.info(
        f"{process_props.rank}/{process_props.comm_size}: size of computed field {result.asnumpy().shape}"
    )
    exchange.exchange(dims.CellDim, result, stream=definitions.BLOCK)

    assert test_helpers.dallclose(result.asnumpy(), field_ref.asnumpy())


inputs_ls = [[2.0, 2.0, 4.0, 1.0], [2.0, 1.0], [30.0], [], [-10, 20, 4]]


@pytest.mark.parametrize("global_list", inputs_ls)
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_global_reductions_min(
    process_props: definitions.ProcessProperties,
    backend_like: model_backends.BackendLike,
    global_list: list[data_alloc.ScalarT],
) -> None:
    my_rank = process_props.rank
    xp = data_alloc.import_array_ns(model_backends.get_allocator(backend_like))
    comm_size = process_props.comm_size
    chunks = np.array_split(global_list, comm_size)
    local_data = xp.array(chunks[my_rank])

    global_reduc = definitions.create_reduction(process_props)

    if len(global_list) > 0:
        min_val = global_reduc.min(local_data, array_ns=xp)
        expected_val = np.min(global_list)
        assert expected_val == min_val
    else:
        with pytest.raises(ValueError, match="global_min requires a non-empty buffer"):
            global_reduc.min(local_data, array_ns=xp)


@pytest.mark.parametrize("global_list", inputs_ls)
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_global_reductions_max(
    process_props: definitions.ProcessProperties,
    backend_like: model_backends.BackendLike,
    global_list: list[data_alloc.ScalarT],
) -> None:
    my_rank = process_props.rank
    xp = data_alloc.import_array_ns(model_backends.get_allocator(backend_like))
    comm_size = process_props.comm_size
    chunks = np.array_split(global_list, comm_size)
    local_data = xp.array(chunks[my_rank])

    global_reduc = definitions.create_reduction(process_props)

    if len(global_list) > 0:
        max_val = global_reduc.max(local_data, array_ns=xp)
        expected_val = np.max(global_list)
        assert expected_val == max_val
    else:
        with pytest.raises(ValueError, match="global_max requires a non-empty buffer"):
            global_reduc.max(local_data, array_ns=xp)


@pytest.mark.parametrize("global_list", inputs_ls)
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_global_reductions_sum(
    process_props: definitions.ProcessProperties,
    backend_like: model_backends.BackendLike,
    global_list: list[data_alloc.ScalarT],
) -> None:
    my_rank = process_props.rank
    xp = data_alloc.import_array_ns(model_backends.get_allocator(backend_like))
    comm_size = process_props.comm_size
    chunks = np.array_split(global_list, comm_size)
    local_data = xp.array(chunks[my_rank])

    global_reduc = definitions.create_reduction(process_props)

    if len(global_list) > 0:
        sum_val = global_reduc.sum(local_data, array_ns=xp)
        expected_val = np.sum(global_list)
        assert expected_val == sum_val
    else:
        with pytest.raises(ValueError, match="global_sum requires a non-empty buffer"):
            global_reduc.sum(local_data, array_ns=xp)


@pytest.mark.parametrize("global_list", inputs_ls)
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_global_reductions_mean(
    process_props: definitions.ProcessProperties,
    backend_like: model_backends.BackendLike,
    global_list: list[data_alloc.ScalarT],
) -> None:
    my_rank = process_props.rank
    xp = data_alloc.import_array_ns(model_backends.get_allocator(backend_like))
    comm_size = process_props.comm_size
    chunks = np.array_split(global_list, comm_size)
    local_data = xp.array(chunks[my_rank])
    global_reduc = definitions.create_reduction(process_props)

    if len(global_list) > 0:
        mean_val = global_reduc.mean(local_data, array_ns=xp)
        expected_val = np.mean(global_list)
        assert expected_val == mean_val
    else:
        with pytest.raises(ValueError, match="global_mean requires a non-empty buffer"):
            global_reduc.mean(local_data, array_ns=xp)
