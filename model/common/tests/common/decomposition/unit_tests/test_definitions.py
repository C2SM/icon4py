# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses

import numpy as np
import pytest
from gt4py.next import common as gtx_common

import icon4py.model.common.dimension as dims
import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.common.decomposition import definitions, halo
from icon4py.model.common.grid import simple

from . import utils
from .mpi_tests.test_halo import assert_same_entries
from icon4py.model.common.decomposition.definitions import (
    DecompositionInfo,
    SingleNodeExchange,
    create_exchange,
)
from icon4py.model.testing.fixtures.datatest import (  # import fixtures form test_utils
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)


@pytest.mark.datatest
def test_create_single_node_runtime_without_mpi(icon_grid, processor_props):  # fixture
    decomposition_info = definitions.DecompositionInfo(
        klevels=10,
    )
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    assert isinstance(exchange, definitions.SingleNodeExchange)


@dataclasses.dataclass(frozen=True)
class DummyProps(definitions.ProcessProperties):
    def __init__(self, rank: int):
        object.__setattr__(self, "rank", rank % 4)
        object.__setattr__(self, "comm", None)
        object.__setattr__(self, "comm_name", "dummy on 4")
        object.__setattr__(self, "comm_size", 4)


def dummy_four_ranks(rank) -> definitions.ProcessProperties:
    return DummyProps(rank=rank)


def get_neighbor_tables_for_simple_grid() -> dict[str, data_alloc.NDArray]:
    grid = simple.simple_grid()
    neighbor_tables = {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }
    return neighbor_tables


offsets = [dims.E2C, dims.E2V, dims.C2E, dims.C2E2C, dims.V2C, dims.V2E, dims.C2V, dims.E2C2V]


@pytest.mark.parametrize("offset", offsets)
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_global_to_local_index(offset, rank):
    grid = simple.simple_grid()
    neighbor_tables = {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }
    props = dummy_four_ranks(rank)
    halo_constructor = halo.IconLikeHaloConstructor(props, neighbor_tables, 1)
    decomposition_info = halo_constructor(utils.SIMPLE_DISTRIBUTION)
    source_indices_on_local_grid = decomposition_info.global_index(offset.target[0])

    offset_full_grid = grid.connectivities[offset.value].ndarray[source_indices_on_local_grid]
    neighbor_dim = offset.source
    neighbor_index_full_grid = decomposition_info.global_index(neighbor_dim)

    local_offset = decomposition_info.global_to_local(neighbor_dim, offset_full_grid)

    ## assert by backmapping

    for i in range(local_offset.shape[0]):
        for k in range(local_offset.shape[1]):
            k_ = local_offset[i][k]
            if k_ == -1:
                # global index is not on this local patch:
                assert not np.isin(offset_full_grid[i][k], neighbor_index_full_grid)
            else:
                (
                    neighbor_index_full_grid[k_] == offset_full_grid[i][k],
                    f"failed to map [{offset_full_grid[i]}] to local: [{local_offset[i]}]",
                )


# TODO this duplicates and serializes a test from mpi_tests/test_halo.py
@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_halo_constructor_decomposition_info_global_indices(dim, rank):
    simple_neighbor_tables = get_neighbor_tables_for_simple_grid()
    props = dummy_four_ranks(rank)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=props,
        num_levels=1,
    )

    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    my_halo = decomp_info.global_index(dim, definitions.DecompositionInfo.EntryType.HALO)
    print(f"rank {props.rank} has halo {dim} : {my_halo}")
    assert my_halo.size == len(utils.HALO[dim][props.rank])
    assert np.setdiff1d(my_halo, utils.HALO[dim][props.rank], assume_unique=True).size == 0
    my_owned = decomp_info.global_index(dim, definitions.DecompositionInfo.EntryType.OWNED)
    print(f"rank {props.rank} owns {dim} : {my_owned} ")
    assert_same_entries(dim, my_owned, utils.OWNED, props.rank)
