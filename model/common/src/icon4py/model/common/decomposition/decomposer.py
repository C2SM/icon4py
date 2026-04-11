# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar, Protocol, runtime_checkable

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

    ICON grids are built from an icosahedron (20 triangular faces). The faces
    are ordered in 4 latitude rings of 5: north polar (0-4), north mid-lat
    (5-9), south mid-lat (10-14), south polar (15-19). Adjacent faces pair
    into 10 diamonds:
      - d=0..4: face d + face d+5  (north polar + north mid-lat)
      - d=5..9: face d+5 + face d+10  (south mid-lat + south polar)

    With root R, each face has R² sub-triangles, giving 10 diamonds of 2*R²
    sub-triangles each. With bisection level B, each sub-triangle contains
    4^B cells in contiguous index order following a recursive quad-tree.

    The decomposer assigns groups of diamonds (or sub-blocks within diamonds)
    to ranks. At the coarsest level, 10 ranks get one diamond each. For more
    ranks, the quad-tree subdivision within each sub-triangle is used.

    Supported rank counts: any divisor of 10 * 2*R² * 4^k for k=0,...,B.
    """

    N_DIAMONDS: ClassVar[int] = 10
    DIAMOND_FACES: ClassVar[list[tuple[int, int]]] = [
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),  # north
        (10, 15),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),  # south
    ]

    def __init__(self, grid_root: int, grid_level: int):
        self._grid_root = grid_root
        self._grid_level = grid_level

    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        import numpy as np

        n_cells = adjacency_matrix.shape[0]
        R = self._grid_root
        B = self._grid_level
        sub_tris_per_face = R**2
        cells_per_sub_tri = 4**B
        cells_per_face = sub_tris_per_face * cells_per_sub_tri

        # Find the smallest subdivision level k such that
        # the total number of sub-blocks is divisible by num_partitions.
        # At level k, each diamond has 2 * R² * 4^k sub-blocks.
        for k in range(B + 1):
            sub_blocks_per_diamond = 2 * sub_tris_per_face * (4**k)
            total_sub_blocks = self.N_DIAMONDS * sub_blocks_per_diamond
            if total_sub_blocks % num_partitions == 0:
                break
        else:
            valid = set()
            for kk in range(B + 1):
                nsb = self.N_DIAMONDS * 2 * sub_tris_per_face * (4**kk)
                valid.update(d for d in range(1, nsb + 1) if nsb % d == 0)
            valid_sorted = sorted(valid)
            raise ValueError(
                f"StructuredDecomposer: {num_partitions} ranks cannot evenly divide "
                f"the grid (root={R}, level={B}, {self.N_DIAMONDS} diamonds). "
                f"Supported rank counts (up to {total_sub_blocks}): "
                f"{valid_sorted[:20]}{'...' if len(valid_sorted) > 20 else ''}"
            )

        sub_block_size = cells_per_sub_tri // (4**k)
        blocks_per_rank = total_sub_blocks // num_partitions

        partition = np.empty(n_cells, dtype=np.int32)
        global_sub_block = 0

        for face_a, face_b in self.DIAMOND_FACES:
            for face in (face_a, face_b):
                face_start = face * cells_per_face
                for st in range(sub_tris_per_face):
                    tri_start = face_start + st * cells_per_sub_tri
                    for sb in range(4**k):
                        cell_start = tri_start + sb * sub_block_size
                        rank = global_sub_block // blocks_per_rank
                        partition[cell_start : cell_start + sub_block_size] = rank
                        global_sub_block += 1

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
