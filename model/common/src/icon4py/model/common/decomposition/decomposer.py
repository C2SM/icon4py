# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Protocol, runtime_checkable

import gt4py.next as gtx
import numpy as np

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

        import pymetis  # noqa: PLC0415 [import-outside-top-level]

        # METIS only runs on CPU. Invalid entries denote physical boundaries and are excluded.
        adjacency_matrix_np = data_alloc.as_numpy(adjacency_matrix)
        mask = adjacency_matrix_np >= 0
        counts = mask.sum(axis=1)
        xadj = np.zeros(len(counts) + 1, dtype=np.int32)
        xadj[1:] = np.cumsum(counts)
        adjncy = adjacency_matrix_np[mask]

        # The partitioning is done on all ranks, and this assumes that the partitioning is deterministic.
        _, partition_index = pymetis.part_graph(
            nparts=num_partitions,
            adjacency=pymetis.CSRAdjacency(xadj, adjncy),
        )
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
