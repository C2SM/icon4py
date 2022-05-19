from dataclasses import dataclass

import numpy as np
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
)

from src.icon4py.dimension import C2EDim, EdgeDim, CellDim, KDim


# periodic
#
# 0v---0e-- 1v---3e-- 2v---6e-- 0v
# |  \ 0c   |  \ 1c   |  \
# |   \1e   |   \4e   |   \7e
# |2e   \   |5e   \   |8e   \
# |      \  |      \  |      \
# 3v---9e-- 4v--12e-- 5v--15e-- 3v
# |  \ 2c   |  \ 3c   |  \
# |   \10e  |   \13e  |   \16e
# |11e  \   |14e  \   |17e  \
# |      \  |      \  |      \
# 6v--18e-- 7v--21e-- 8v--24e-- 6v
# |  \      |  \      |  \
# |   \19e  |   \22e  |   \25e
# |20e  \   |23e  \   |26e  \
# |      \  |      \  |      \
# 0v       1v         2v        0v


@dataclass
class SimpleMeshData:
    c2e_table = np.asarray(
        [
            [0, 1, 5],
            [3, 4, 8],
            [6, 7, 2],
            [1, 2, 9],
            [4, 5, 12],
            [7, 8, 15],
            [9, 10, 14],
            [12, 13, 17],
            [15, 16, 11],
            [10, 11, 18],
            [13, 14, 21],
            [16, 17, 24],
            [18, 19, 23],
            [21, 22, 26],
            [24, 25, 20],
            [19, 20, 0],
            [22, 23, 3],
            [25, 26, 6],
        ]
    )
    diamond_table = np.asarray(
        [
            [0, 1, 4, 6],  # 0
            [0, 4, 1, 3],
            [0, 3, 4, 2],
            [1, 2, 5, 7],  # 3
            [1, 5, 2, 4],
            [1, 4, 5, 0],
            [2, 0, 3, 8],  # 6
            [2, 3, 0, 5],
            [2, 5, 1, 3],
            [3, 4, 0, 7],  # 9
            [3, 7, 4, 6],
            [3, 6, 5, 7],
            [4, 5, 1, 8],  # 12
            [4, 8, 5, 7],
            [4, 7, 3, 8],
            [5, 3, 2, 6],  # 15
            [5, 6, 3, 8],
            [5, 8, 4, 6],
            [6, 7, 3, 1],  # 18
            [6, 1, 7, 0],
            [6, 0, 1, 8],
            [7, 8, 4, 2],  # 21
            [7, 2, 8, 1],
            [7, 1, 6, 2],
            [8, 6, 5, 0],  # 24
            [8, 0, 6, 2],
            [8, 2, 7, 0],
        ]
    )


class SimpleMesh:
    _DEFAULT_K_LEVEL = 10

    def __init__(self, k_level: int = _DEFAULT_K_LEVEL):
        self.diamond_arr = SimpleMeshData.diamond_table
        self.c2e = SimpleMeshData.c2e_table
        self.n_cells = self.c2e.shape[0]
        self.n_c2e = self.c2e.shape[1]
        self.n_edges = 27
        self.n_vertices = 9
        self.k_level = k_level
        self.size = {
            CellDim: self.n_cells,
            EdgeDim: self.n_edges,
            C2EDim: self.n_c2e,
            KDim: self.k_level,
        }

    def get_c2e_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.c2e, CellDim, EdgeDim, self.n_c2e)

    def get_c2k_offset_provider(self, c2k: np.array) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(c2k, CellDim, EdgeDim, self.k_level)
