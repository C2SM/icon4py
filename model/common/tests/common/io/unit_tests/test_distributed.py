# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import xarray as xr

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base
from icon4py.model.common.io import distributed


NUM_LEVELS = 3

#: synthetic single-rank decomposition with fake halo entries and permuted global
#: indices: local entries 2 and 5 are halos; the owned entries sit at the global
#: positions [2, 0, 3, 1] (a permutation, as required of a decomposition)
CELL_OWNER_MASK = np.asarray([True, True, False, True, True, False])
CELL_GLOBAL_INDEX = np.asarray([2, 0, 1, 3, 1, 0], dtype=np.int64)
EDGE_OWNER_MASK = np.asarray([True, True, True, False])
EDGE_GLOBAL_INDEX = np.asarray([1, 2, 0, 0], dtype=np.int64)
VERTEX_OWNER_MASK = np.asarray([True, True])
VERTEX_GLOBAL_INDEX = np.asarray([0, 1], dtype=np.int64)


def synthetic_decomposition_info() -> decomposition.DecompositionInfo:
    info = decomposition.DecompositionInfo()
    for dim, global_index, owner_mask in (
        (dims.CellDim, CELL_GLOBAL_INDEX, CELL_OWNER_MASK),
        (dims.EdgeDim, EDGE_GLOBAL_INDEX, EDGE_OWNER_MASK),
        (dims.VertexDim, VERTEX_GLOBAL_INDEX, VERTEX_OWNER_MASK),
    ):
        halo_levels = np.where(
            owner_mask,
            decomposition.DecompositionFlag.OWNED.value,
            decomposition.DecompositionFlag.FIRST_HALO_LEVEL.value,
        )
        info.set_dimension(dim, global_index, owner_mask, halo_levels)
    return info


def cell_field(num_cells: int) -> xr.DataArray:
    values = np.arange(num_cells * NUM_LEVELS, dtype=np.float64).reshape(num_cells, NUM_LEVELS)
    return xr.DataArray(values, dims=("cell", "level"), attrs={"units": "1"})


def test_single_node_distribution_passes_state_through() -> None:
    horizontal_size = base.HorizontalGridSize(num_cells=6, num_edges=4, num_vertices=2)
    distribution = distributed.SingleNodeDistribution(horizontal_size)
    state = {"air_density": cell_field(6)}
    assert distribution.writes_output
    assert distribution.rank_blocks is None
    assert distribution.output_horizontal_size == horizontal_size
    assert distribution.prepare(state) is state


def test_gather_distribution_single_rank_reorders_owned_entries() -> None:
    distribution = distributed.GatherDistribution(
        decomposition.SingleNodeProcessProperties(), synthetic_decomposition_info()
    )
    field = cell_field(CELL_OWNER_MASK.shape[0])
    prepared = distribution.prepare({"air_density": field})

    assert distribution.writes_output
    assert prepared is not None
    gathered = prepared["air_density"]
    assert gathered.dims == field.dims
    assert gathered.shape == (4, NUM_LEVELS)
    # every owned local entry must sit at its global position; halos must be gone
    owned_local_indices = np.flatnonzero(CELL_OWNER_MASK)
    for local_index in owned_local_indices:
        global_position = CELL_GLOBAL_INDEX[local_index]
        assert np.all(gathered.values[global_position] == field.values[local_index])


def test_gather_distribution_single_rank_global_sizes_are_owned_counts() -> None:
    distribution = distributed.GatherDistribution(
        decomposition.SingleNodeProcessProperties(), synthetic_decomposition_info()
    )
    horizontal_size = distribution.output_horizontal_size
    assert horizontal_size.num_cells == int(CELL_OWNER_MASK.sum())
    assert horizontal_size.num_edges == int(EDGE_OWNER_MASK.sum())
    assert horizontal_size.num_vertices == int(VERTEX_OWNER_MASK.sum())


def test_gather_distribution_rejects_unknown_leading_dimension() -> None:
    distribution = distributed.GatherDistribution(
        decomposition.SingleNodeProcessProperties(), synthetic_decomposition_info()
    )
    bogus = xr.DataArray(np.zeros((2, NUM_LEVELS)), dims=("level", "cell"))
    with pytest.raises(ValueError, match="leading dimension 'level'"):
        distribution.prepare({"air_density": bogus})


def test_rank_block_distribution_single_rank_layout() -> None:
    distribution = distributed.RankBlockDistribution(
        decomposition.SingleNodeProcessProperties(), synthetic_decomposition_info()
    )
    assert distribution.writes_output
    blocks = distribution.rank_blocks
    cell_block = blocks["cell"]
    num_owned_cells = int(CELL_OWNER_MASK.sum())
    assert cell_block.start == 0
    assert cell_block.count == num_owned_cells
    assert cell_block.size == num_owned_cells
    assert cell_block.padded_size == num_owned_cells
    assert cell_block.global_size == num_owned_cells
    assert np.all(cell_block.global_index == CELL_GLOBAL_INDEX[CELL_OWNER_MASK])
    assert distribution.output_horizontal_size.num_cells == num_owned_cells


def test_rank_block_distribution_block_alignment_rounds_block_size_up() -> None:
    # block size rounds up to a multiple of the alignment; count and global size are
    # unaffected
    distribution = distributed.RankBlockDistribution(
        decomposition.SingleNodeProcessProperties(),
        synthetic_decomposition_info(),
        block_alignment=3,
    )
    num_owned_cells = int(CELL_OWNER_MASK.sum())  # 4 -> block size 6
    cell_block = distribution.rank_blocks["cell"]
    assert cell_block.count == num_owned_cells
    assert cell_block.size == 6
    assert cell_block.padded_size == 6
    assert cell_block.global_size == num_owned_cells
    # 3 owned edges are already aligned; 2 owned vertices round up to 3
    assert distribution.rank_blocks["edge"].size == 3
    assert distribution.rank_blocks["vertex"].size == 3
    assert distribution.output_horizontal_size == base.HorizontalGridSize(
        num_cells=6, num_edges=3, num_vertices=3
    )


def test_rank_block_distribution_prepare_strips_halos() -> None:
    distribution = distributed.RankBlockDistribution(
        decomposition.SingleNodeProcessProperties(), synthetic_decomposition_info()
    )
    field = cell_field(CELL_OWNER_MASK.shape[0])
    prepared = distribution.prepare({"air_density": field})
    assert prepared is not None
    stripped = prepared["air_density"]
    assert stripped.shape == (int(CELL_OWNER_MASK.sum()), NUM_LEVELS)
    assert np.all(stripped.values == field.values[CELL_OWNER_MASK])
    # the stripped data is a copy, detached from the live model state
    field.values[CELL_OWNER_MASK] += 1.0
    assert np.all(stripped.values == field.values[CELL_OWNER_MASK] - 1.0)


def _non_partition_decomposition_info() -> decomposition.DecompositionInfo:
    # duplicate owned global indices: the owned entries of the ranks are no permutation
    # of the global grid, which would silently corrupt the reassembled fields
    info = decomposition.DecompositionInfo()
    for dim in (dims.CellDim, dims.EdgeDim, dims.VertexDim):
        owner_mask = np.ones(3, dtype=bool)
        halo_levels = np.full(3, decomposition.DecompositionFlag.OWNED.value)
        info.set_dimension(dim, np.asarray([0, 0, 1], dtype=np.int64), owner_mask, halo_levels)
    return info


def test_gather_distribution_rejects_non_partition_owned_indices() -> None:
    with pytest.raises(ValueError, match="do not partition"):
        distributed.GatherDistribution(
            decomposition.SingleNodeProcessProperties(), _non_partition_decomposition_info()
        )


def test_rank_block_distribution_rejects_non_partition_owned_indices() -> None:
    with pytest.raises(ValueError, match="do not partition"):
        distributed.RankBlockDistribution(
            decomposition.SingleNodeProcessProperties(), _non_partition_decomposition_info()
        )


@pytest.mark.parametrize("block_alignment", [0, -4])
def test_rank_block_distribution_rejects_non_positive_block_alignment(
    block_alignment: int,
) -> None:
    with pytest.raises(ValueError, match="Invalid block alignment"):
        distributed.RankBlockDistribution(
            decomposition.SingleNodeProcessProperties(),
            synthetic_decomposition_info(),
            block_alignment=block_alignment,
        )


def test_rank_block_global_index_is_read_only() -> None:
    distribution = distributed.RankBlockDistribution(
        decomposition.SingleNodeProcessProperties(), synthetic_decomposition_info()
    )
    cell_block = distribution.rank_blocks["cell"]
    with pytest.raises(ValueError, match="read-only"):
        cell_block.global_index[0] = 99
