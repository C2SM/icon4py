# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common import dimension as dims, exceptions, model_backends
from icon4py.model.common.decomposition import definitions, halo
from icon4py.model.common.grid import base as base_grid

from ...fixtures import backend_like, processor_props
from .. import utils
from ..fixtures import simple_neighbor_tables


@pytest.mark.parametrize("rank", [0, 1, 2, 4])
def test_halo_constructor_owned_cells(rank, simple_neighbor_tables, backend_like):
    processor_props = utils.DummyProps(rank=rank)
    allocator = model_backends.get_allocator(backend_like)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        allocator=allocator,
    )
    my_owned_cells = halo_generator.owned_cells(utils.SIMPLE_DISTRIBUTION)

    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(utils._CELL_OWN[processor_props.rank])
    assert np.setdiff1d(my_owned_cells, utils._CELL_OWN[processor_props.rank]).size == 0


@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
@pytest.mark.parametrize("rank", [0, 1, 2, 4])
def test_halo_constructor_decomposition_info_global_indices(rank, simple_neighbor_tables, dim):
    processor_props = utils.DummyProps(rank=rank)
    if processor_props.comm_size != 4:
        pytest.skip(
            f"This test requires exactly 4 MPI ranks, current run has {processor_props.comm_size}"
        )

    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
    )

    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    my_halo = decomp_info.global_index(dim, definitions.DecompositionInfo.EntryType.HALO)
    print(f"rank {processor_props.rank} has halo {dim} : {my_halo}")
    expected = len(utils.HALO[dim][processor_props.rank])
    assert (
        my_halo.size == expected
    ), f"total halo size does not match for dim {dim}- expected {expected} bot was {my_halo.size}"
    assert (
        missing := np.setdiff1d(
            my_halo, utils.HALO[dim][processor_props.rank], assume_unique=True
        ).size
        == 0
    ), f"missing halo elements are {missing}"
    my_owned = decomp_info.global_index(dim, definitions.DecompositionInfo.EntryType.OWNED)
    print(f"rank {processor_props.rank} owns {dim} : {my_owned} ")
    utils.assert_same_entries(dim, my_owned, utils.OWNED, processor_props.rank)


@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_halo_constructor_decomposition_info_halo_levels(rank, dim, simple_neighbor_tables):
    processor_props = utils.DummyProps(rank=rank)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
    )
    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    my_halo_levels = decomp_info.halo_levels(dim)
    print(f"{dim.value}: rank {processor_props.rank} has halo levels {my_halo_levels} ")
    assert np.all(
        my_halo_levels != definitions.DecompositionFlag.UNDEFINED
    ), "All indices should have a defined DecompositionFlag"

    assert np.where(my_halo_levels == definitions.DecompositionFlag.OWNED)[0].size == len(
        utils.OWNED[dim][processor_props.rank]
    )
    owned_local_indices = decomp_info.local_index(
        dim, definitions.DecompositionInfo.EntryType.OWNED
    )
    assert np.all(
        my_halo_levels[owned_local_indices] == definitions.DecompositionFlag.OWNED
    ), "owned local indices should have DecompositionFlag.OWNED"
    first_halo_level_local_index = np.where(
        my_halo_levels == definitions.DecompositionFlag.FIRST_HALO_LEVEL
    )[0]
    first_halo_level_global_index = decomp_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.ALL
    )[first_halo_level_local_index]
    utils.assert_same_entries(
        dim, first_halo_level_global_index, utils.FIRST_HALO_LINE, processor_props.rank
    )
    second_halo_level_local_index = np.where(
        my_halo_levels == definitions.DecompositionFlag.SECOND_HALO_LEVEL
    )[0]
    second_halo_level_global_index = decomp_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.ALL
    )[second_halo_level_local_index]
    utils.assert_same_entries(
        dim, second_halo_level_global_index, utils.SECOND_HALO_LINE, processor_props.rank
    )
    third_halo_level_index = np.where(
        my_halo_levels == definitions.DecompositionFlag.THIRD_HALO_LEVEL
    )[0]
    third_halo_level_global_index = decomp_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.ALL
    )[third_halo_level_index]
    utils.assert_same_entries(
        dim, third_halo_level_global_index, utils.THIRD_HALO_INE, processor_props.rank
    )


def test_no_halo():
    grid_size = base_grid.HorizontalGridSize(num_cells=9, num_edges=14, num_vertices=6)
    halo_generator = halo.NoHalos(horizontal_size=grid_size, allocator=None)
    decomposition = halo.SingleNodeDecomposer()
    decomposition_info = halo_generator(decomposition(np.arange(grid_size.num_cells), 1))
    # cells
    np.testing.assert_allclose(
        np.arange(grid_size.num_cells), decomposition_info.global_index(dims.CellDim)
    )
    assert np.all(decomposition_info.owner_mask(dims.CellDim))
    assert np.all(
        decomposition_info.halo_levels(dims.CellDim) == definitions.DecompositionFlag.OWNED
    )
    # edges
    np.testing.assert_allclose(
        np.arange(grid_size.num_edges), decomposition_info.global_index(dims.EdgeDim)
    )
    assert np.all(
        decomposition_info.halo_levels(dims.EdgeDim) == definitions.DecompositionFlag.OWNED
    )
    assert np.all(decomposition_info.owner_mask(dims.EdgeDim))
    # vertices
    np.testing.assert_allclose(
        np.arange(grid_size.num_vertices), decomposition_info.global_index(dims.VertexDim)
    )
    assert np.all(
        decomposition_info.halo_levels(dims.VertexDim) == definitions.DecompositionFlag.OWNED
    )
    assert np.all(decomposition_info.owner_mask(dims.VertexDim))


def test_halo_constructor_validate_rank_mapping_wrong_shape(simple_neighbor_tables):
    processor_props = utils.DummyProps(rank=2)
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    with pytest.raises(exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            run_properties=processor_props,
        )
        halo_generator(np.zeros((num_cells, 3), dtype=int))
    assert f"should have shape ({num_cells},)" in e.value.args[0]


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_halo_constructor_validate_number_of_node_mismatch(rank, simple_neighbor_tables):
    processor_props = utils.DummyProps(rank=rank)
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    distribution = (processor_props.comm_size + 1) * np.ones((num_cells,), dtype=int)
    with pytest.raises(expected_exception=exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            run_properties=processor_props,
        )
        halo_generator(distribution)
    assert "The distribution assumes more nodes than the current run" in e.value.args[0]
