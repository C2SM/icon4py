# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Protocol, runtime_checkable

import gt4py.next as gtx

from icon4py.model.common.utils import data_allocation as data_alloc


@runtime_checkable
class Decomposer(Protocol):
    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """
        Call the decomposition.

        Args:
            adjacency_matrix: cell-to-cell connectivity matrix on the global (undecomposed) grid. In the Icon4py context this C2E2C
            n_part: number of nodes
        """
        ...


class MetisDecomposer(Decomposer):
    """
    A simple decomposer using METIS for partitioning a grid topology.

    We use the simple pythonic interface to pymetis: just passing the adjacency matrix, which for ICON is
    the full grid C2E2C neigbhor table.
    if more control is needed (for example by using weights we need to switch to the C like interface)
    https://documen.tician.de/pymetis/functionality.html
    """

    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """
        Generate partition labels for this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Args:
            n_part: int, number of partitions to create
            adjacency_matrix: nd array: neighbor table describing of the main dimension object to be distributed: for example cell -> cell neighbors
        Returns: data_alloc.NDArray: array with partition label (int, rank number) for each cell
        """

        import pymetis  # type: ignore [import-untyped]

        # Invalid indices are not allowed here. Metis will segfault or fail if
        # there are any invalid indices in the adjacency matrix.
        assert (adjacency_matrix >= 0).all()

        # The partitioning is done on all ranks, and this assumes that the
        # partitioning is deterministic.
        _, partition_index = pymetis.part_graph(nparts=num_partitions, adjacency=adjacency_matrix)
        return data_alloc.array_namespace(adjacency_matrix).array(partition_index)


class StructuredDecomposer(Decomposer):
    """
    A decomposer that exploits the structured cell ordering of ICON grids.

    ICON grids with root R and bisection level B have 20*R² base diamond blocks,
    each containing 4^B cells in contiguous index order. The cells within each
    block follow a recursive quad-tree subdivision, so blocks can be further
    split into sub-blocks of 4^k cells.

    This decomposer assigns contiguous groups of (sub-)blocks to ranks,
    avoiding the expensive METIS graph partitioning entirely.

    Supported rank counts: any divisor of 20*R² * 4^k for k=0,1,...,B.
    """

    def __init__(self, grid_root: int, grid_level: int):
        self._grid_root = grid_root
        self._grid_level = grid_level

    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        import numpy as np

        n_cells = adjacency_matrix.shape[0]
        n_base_blocks = 20 * self._grid_root**2

        # Find the smallest subdivision level k such that
        # the number of sub-blocks is divisible by num_partitions
        for k in range(self._grid_level + 1):
            n_sub_blocks = n_base_blocks * (4**k)
            if n_sub_blocks % num_partitions == 0:
                break
        else:
            valid = []
            for kk in range(self._grid_level + 1):
                nsb = n_base_blocks * (4**kk)
                valid.extend(d for d in range(1, nsb + 1) if nsb % d == 0)
            valid = sorted(set(valid))
            raise ValueError(
                f"StructuredDecomposer: {num_partitions} ranks cannot evenly divide "
                f"the grid (root={self._grid_root}, level={self._grid_level}, "
                f"{n_base_blocks} base blocks). "
                f"Supported rank counts (up to {n_base_blocks * 4**self._grid_level}): "
                f"{valid[:20]}{'...' if len(valid) > 20 else ''}"
            )

        sub_block_size = n_cells // n_sub_blocks
        blocks_per_rank = n_sub_blocks // num_partitions

        # Assign each cell to a rank based on its sub-block index
        cell_indices = np.arange(n_cells, dtype=np.int32)
        sub_block_ids = cell_indices // sub_block_size
        partition = sub_block_ids // blocks_per_rank

        return data_alloc.array_namespace(adjacency_matrix).asarray(partition)


class SingleNodeDecomposer(Decomposer):
    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """Dummy decomposer for single node: assigns all cells to rank = 0"""
        if num_partitions != 1:
            raise ValueError(
                f"SingleNodeDecomposer can only be used for num_partitions=1, but got {num_partitions}"
            )

        return data_alloc.array_namespace(adjacency_matrix).zeros(
            adjacency_matrix.shape[0],
            dtype=gtx.int32,
        )
