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

        Uses the CSR (xadj/adjncy) interface instead of the Pythonic adjacency
        interface to avoid slow Python list-of-lists conversion for large grids.

        Args:
            num_partitions: int, number of partitions to create
            adjacency_matrix: nd array: neighbor table describing of the main dimension object to be distributed: for example cell -> cell neighbors
        Returns: data_alloc.NDArray: array with partition label (int, rank number) for each cell
        """
        import numpy as np
        import pymetis  # type: ignore [import-untyped]

        # Invalid indices are not allowed here. Metis will segfault or fail if
        # there are any invalid indices in the adjacency matrix.
        adj = np.asarray(adjacency_matrix)
        assert (adj >= 0).all()

        # Build CSR format (xadj/adjncy) from the dense adjacency matrix.
        # For C2E2C the matrix is (n_cells, n_neighbors_per_cell) with fixed width.
        n_nodes, n_neighbors = adj.shape
        xadj = np.arange(0, (n_nodes + 1) * n_neighbors, n_neighbors, dtype=np.int32)
        adjncy = adj.ravel().astype(np.int32)

        # The partitioning is done on all ranks, and this assumes that the
        # partitioning is deterministic.
        _, partition_index = pymetis.part_graph(nparts=num_partitions, xadj=xadj, adjncy=adjncy)
        return data_alloc.array_namespace(adjacency_matrix).array(partition_index)


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
