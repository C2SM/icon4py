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

    Within each face (for R=2^m), cells follow a recursive quad-tree where
    each triangle splits into 4 children. The adjacency pattern at every
    level is: child 2 is the center (adjacent to 0, 1, 3), while 0, 1, 3
    are corners adjacent only to 2.

    A diamond (pair of adjacent triangles A, B) can be recursively split
    into 4 sub-diamonds using this pattern:
      - (A0, A2): intra-A diamond (corner + center)
      - (A1, B3): cross diamond
      - (A3, B1): cross diamond
      - (B0, B2): intra-B diamond (corner + center)

    This gives 10 * 4^d sub-diamonds at depth d, all with compact
    diamond shapes. Supported rank counts: any divisor of 10 * 4^d
    for d=0,...,max_depth.
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

    @staticmethod
    def _quad_depth(grid_root: int) -> int:
        """Number of quad-tree levels in R² sub-triangles (R must be a power of 2)."""
        r = grid_root
        depth = 0
        while r > 1:
            if r % 2 != 0:
                raise ValueError(
                    f"Recursive diamond decomposition requires grid_root to be a "
                    f"power of 2, got {grid_root}"
                )
            r //= 2
            depth += 1
        return depth

    @staticmethod
    def _diamond_leaves(a_start, a_size, b_start, b_size, depth):
        """
        Yield sub-diamond leaves in recursive diamond order.

        Each leaf is (tri_a_start, tri_a_size, tri_b_start, tri_b_size).
        At depth 0, yields the diamond itself. At depth d, yields 4^d sub-diamonds.
        """
        if depth == 0:
            yield (a_start, a_size, b_start, b_size)
            return

        q = a_size // 4
        a0, a1, a2, a3 = a_start, a_start + q, a_start + 2 * q, a_start + 3 * q
        b0, b1, b2, b3 = b_start, b_start + q, b_start + 2 * q, b_start + 3 * q

        yield from StructuredDecomposer._diamond_leaves(a0, q, a2, q, depth - 1)
        yield from StructuredDecomposer._diamond_leaves(a1, q, b3, q, depth - 1)
        yield from StructuredDecomposer._diamond_leaves(a3, q, b1, q, depth - 1)
        yield from StructuredDecomposer._diamond_leaves(b0, q, b2, q, depth - 1)

    def _max_depth(self) -> int:
        return self._quad_depth(self._grid_root) + self._grid_level

    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        import numpy as np

        n_cells = adjacency_matrix.shape[0]
        R = self._grid_root
        B = self._grid_level
        cells_per_face = R**2 * 4**B
        max_depth = self._max_depth()

        # Find minimum depth d such that 10 * 4^d is divisible by num_partitions
        for d in range(max_depth + 1):
            total_sub_diamonds = self.N_DIAMONDS * (4**d)
            if total_sub_diamonds % num_partitions == 0:
                break
        else:
            valid = set()
            for dd in range(max_depth + 1):
                nsub = self.N_DIAMONDS * (4**dd)
                valid.update(div for div in range(1, nsub + 1) if nsub % div == 0)
            valid_sorted = sorted(valid)
            raise ValueError(
                f"StructuredDecomposer: {num_partitions} ranks cannot evenly divide "
                f"the grid (root={R}, level={B}, {self.N_DIAMONDS} diamonds). "
                f"Supported rank counts (up to {total_sub_diamonds}): "
                f"{valid_sorted[:20]}{'...' if len(valid_sorted) > 20 else ''}"
            )

        sub_diamonds_per_rank = total_sub_diamonds // num_partitions

        partition = np.empty(n_cells, dtype=np.int32)
        sub_diamond_idx = 0

        for face_a, face_b in self.DIAMOND_FACES:
            fa_start = face_a * cells_per_face
            fb_start = face_b * cells_per_face
            for ta_start, ta_size, tb_start, tb_size in self._diamond_leaves(
                fa_start, cells_per_face, fb_start, cells_per_face, d
            ):
                rank = sub_diamond_idx // sub_diamonds_per_rank
                partition[ta_start : ta_start + ta_size] = rank
                partition[tb_start : tb_start + tb_size] = rank
                sub_diamond_idx += 1

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
