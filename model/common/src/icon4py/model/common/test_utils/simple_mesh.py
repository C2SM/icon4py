# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass

import numpy as np
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider, StridedNeighborOffsetProvider

from icon4py.model.common.dimension import (
    C2E2C2E2CDim,
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CECDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
    ECDim
)


# periodic
#
# 0v---0e-- 1v---3e-- 2v---6e-- 0v
# |  \ 0c   |  \ 1c   |  \2c
# |   \1e   |   \4e   |   \7e
# |2e   \   |5e   \   |8e   \
# |  3c   \ |   4c  \ |    5c\
# 3v---9e-- 4v--12e-- 5v--15e-- 3v
# |  \ 6c   |  \ 7c   |  \ 8c
# |   \10e  |   \13e  |   \16e
# |11e  \   |14e  \   |17e  \
# |  9c  \  |  10c \  |  11c \
# 6v--18e-- 7v--21e-- 8v--24e-- 6v
# |  \12c   |  \ 13c  |  \ 14c
# |   \19e  |   \22e  |   \25e
# |20e  \   |23e  \   |26e  \
# |  15c  \ | 16c   \ | 17c  \
# 0v       1v         2v        0v


@dataclass
class SimpleMeshData:
    c2v_table = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 5],
            [2, 0, 3],
            [0, 3, 4],
            [1, 4, 5],
            [2, 5, 3],
            [3, 4, 7],
            [4, 5, 8],
            [5, 3, 6],
            [3, 6, 7],
            [4, 7, 8],
            [5, 8, 6],
            [6, 7, 1],
            [7, 8, 2],
            [8, 6, 0],
            [6, 0, 1],
            [7, 1, 2],
            [8, 2, 0],
        ]
    )

    e2c2v_table = np.asarray(
        [
            [0, 1, 4, 6],  # 0
            [0, 4, 1, 3],  # 1
            [0, 3, 4, 2],  # 2
            [1, 2, 5, 7],  # 3
            [1, 5, 2, 4],  # 4
            [1, 4, 5, 0],  # 5
            [2, 0, 3, 8],  # 6
            [2, 3, 5, 0],  # 7
            [2, 5, 1, 3],  # 8
            [3, 4, 0, 7],  # 9
            [3, 7, 4, 6],  # 10
            [3, 6, 7, 5],  # 11
            [4, 5, 8, 1],  # 12
            [4, 8, 7, 5],  # 13
            [4, 7, 3, 8],  # 14
            [5, 3, 6, 2],  # 15
            [6, 5, 3, 8],  # 16
            [8, 5, 6, 4],  # 17
            [6, 7, 3, 1],  # 18
            [6, 1, 7, 0],  # 19
            [6, 0, 1, 8],  # 20
            [7, 8, 2, 4],  # 21
            [7, 2, 8, 1],  # 22
            [7, 1, 2, 6],  # 23
            [8, 6, 0, 5],  # 24
            [8, 0, 6, 2],  # 25
            [8, 2, 0, 6],  # 26
        ]
    )

    e2c_table = np.asarray(
        [
            [0, 15],
            [0, 3],
            [3, 2],
            [1, 16],
            [1, 4],
            [0, 4],
            [2, 17],
            [2, 5],
            [1, 5],
            [3, 6],
            [6, 9],
            [9, 8],
            [4, 7],
            [7, 10],
            [6, 10],
            [5, 8],
            [8, 11],
            [7, 11],
            [9, 12],
            [12, 15],
            [15, 14],
            [10, 13],
            [13, 16],
            [12, 16],
            [11, 14],
            [14, 17],
            [13, 17],
        ]
    )

    e2v_table = np.asarray(
        [
            [0, 1],
            [0, 4],
            [0, 3],
            [1, 2],
            [1, 5],
            [1, 4],
            [2, 0],
            [2, 3],
            [2, 5],
            [3, 4],
            [3, 7],
            [3, 6],
            [4, 5],
            [4, 8],
            [4, 7],
            [5, 3],
            [5, 6],
            [5, 8],
            [6, 7],
            [6, 1],
            [6, 0],
            [7, 8],
            [7, 2],
            [7, 1],
            [8, 6],
            [8, 0],
            [8, 2],
        ]
    )

    e2c2e_table = np.asarray(
        [
            [1, 5, 19, 20],
            [0, 5, 2, 9],
            [1, 9, 6, 7],
            [4, 8, 22, 23],
            [3, 8, 5, 12],
            [0, 1, 4, 12],
            [7, 2, 25, 26],
            [6, 2, 8, 15],
            [3, 4, 7, 15],
            [1, 2, 10, 14],
            [9, 14, 11, 18],
            [10, 18, 15, 16],
            [4, 5, 13, 17],
            [12, 17, 14, 21],
            [9, 10, 13, 21],
            [7, 8, 16, 11],
            [15, 11, 17, 24],
            [12, 13, 16, 24],
            [10, 11, 19, 23],
            [18, 23, 20, 0],
            [19, 0, 24, 25],
            [13, 14, 22, 26],
            [21, 26, 23, 3],
            [18, 19, 22, 3],
            [16, 17, 25, 20],
            [24, 20, 26, 6],
            [25, 6, 21, 22],
        ]
    )

    e2c2eO_table = np.asarray(
        [
            [0, 1, 5, 19, 20],
            [0, 1, 5, 2, 9],
            [1, 2, 9, 6, 7],
            [3, 4, 8, 22, 23],
            [3, 4, 8, 5, 12],
            [0, 1, 5, 4, 12],
            [6, 7, 2, 25, 26],
            [6, 7, 2, 8, 15],
            [3, 4, 8, 7, 15],
            [1, 2, 9, 10, 14],
            [9, 10, 14, 11, 18],
            [10, 11, 18, 15, 16],
            [4, 5, 12, 13, 17],
            [12, 13, 17, 14, 21],
            [9, 10, 14, 13, 21],
            [7, 8, 15, 16, 11],
            [15, 16, 11, 17, 24],
            [12, 13, 17, 16, 24],
            [10, 11, 18, 19, 23],
            [18, 19, 23, 20, 0],
            [19, 20, 0, 24, 25],
            [13, 14, 21, 22, 26],
            [21, 22, 26, 23, 3],
            [18, 19, 23, 22, 3],
            [16, 17, 24, 25, 20],
            [24, 25, 20, 26, 6],
            [25, 26, 6, 21, 22],
        ]
    )

    c2e_table = np.asarray(
        [
            [0, 1, 5],  # cell 0
            [3, 4, 8],  # cell 1
            [6, 7, 2],  # cell 2
            [1, 2, 9],  # cell 3
            [4, 5, 12],  # cell 4
            [7, 8, 15],  # cell 5
            [9, 10, 14],  # cell 6
            [12, 13, 17],  # cell 7
            [15, 16, 11],  # cell 8
            [10, 11, 18],  # cell 9
            [13, 14, 21],  # cell 10
            [16, 17, 24],  # cell 11
            [18, 19, 23],  # cell 12
            [21, 22, 26],  # cell 13
            [24, 25, 20],  # cell 14
            [19, 20, 0],  # cell 15
            [22, 23, 3],  # cell 16
            [25, 26, 6],  # cell 17
        ]
    )

    v2c_table = np.asarray(
        [
            [17, 14, 3, 0, 2, 15],
            [0, 4, 1, 12, 16, 15],
            [1, 5, 2, 16, 13, 17],
            [3, 6, 9, 5, 8, 2],
            [6, 10, 7, 4, 0, 3],
            [7, 11, 8, 5, 1, 4],
            [9, 12, 15, 8, 11, 14],
            [12, 16, 13, 10, 6, 9],
            [13, 17, 14, 11, 7, 10],
        ]
    )

    v2e_table = np.asarray(
        [
            [0, 1, 2, 6, 25, 20],
            [3, 4, 5, 0, 23, 19],
            [6, 7, 8, 3, 22, 26],
            [9, 10, 11, 15, 7, 2],
            [12, 13, 14, 9, 1, 5],
            [15, 16, 17, 12, 4, 8],
            [18, 19, 20, 24, 16, 11],
            [21, 22, 23, 18, 10, 14],
            [24, 25, 26, 21, 13, 17],
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

    c2e2cO_table = np.asarray(
        [
            [15, 4, 3, 0],
            [16, 5, 4, 1],
            [17, 3, 5, 2],
            [0, 6, 2, 3],
            [1, 7, 0, 4],
            [2, 8, 1, 5],
            [3, 10, 9, 6],
            [4, 11, 10, 7],
            [5, 9, 11, 8],
            [6, 12, 8, 9],
            [7, 13, 6, 10],
            [8, 14, 7, 11],
            [9, 16, 15, 12],
            [10, 17, 16, 13],
            [11, 15, 17, 14],
            [12, 0, 14, 15],
            [13, 1, 12, 16],
            [14, 2, 13, 17],
        ]
    )

    c2e2c_table = np.asarray(
        [
            [15, 4, 3],
            [16, 5, 4],
            [17, 3, 5],
            [0, 6, 2],
            [1, 7, 0],
            [2, 8, 1],
            [3, 10, 9],
            [4, 11, 10],
            [5, 9, 11],
            [6, 12, 8],
            [7, 13, 6],
            [8, 14, 7],
            [9, 16, 15],
            [10, 17, 16],
            [11, 15, 17],
            [12, 0, 14],
            [13, 1, 12],
            [14, 2, 13],
        ]
    )

    c2e2c2e2c_table = np.asarray(
        [
            [15, 4, 3, 12, 14, 1, 7, 6, 2],  # 1c
            [16, 5, 4, 12, 13, 2, 8, 7, 0],
            [17, 3, 5, 13, 14, 0, 6, 8, 1],
            [0, 6, 2, 17, 5, 9, 10, 15, 4],
            [1, 7, 0, 15, 3, 16, 5, 10, 11],  # 5c
            [2, 8, 1, 4, 16, 17, 3, 9, 11],
            [3, 10, 9, 2, 0, 7, 13, 8, 12],
            [4, 11, 10, 0, 1, 8, 14, 6, 13],
            [5, 9, 11, 1, 2, 3, 12, 7, 14],
            [6, 12, 8, 5, 11, 3, 10, 16, 15],  # 10c
            [7, 13, 6, 3, 9, 4, 11, 16, 17],
            [8, 14, 7, 4, 10, 5, 9, 15, 17],
            [9, 16, 15, 8, 6, 1, 13, 0, 14],
            [10, 17, 16, 6, 7, 2, 14, 1, 12],
            [11, 15, 17, 7, 8, 2, 13, 0, 12],  # 15c
            [12, 0, 14, 11, 17, 9, 16, 3, 4],
            [13, 1, 12, 9, 15, 10, 17, 4, 5],
            [14, 2, 13, 10, 16, 5, 3, 11, 15],
        ]
    )


class SimpleMesh:
    _DEFAULT_K_LEVEL = 10

    def __init__(self, k_level: int = _DEFAULT_K_LEVEL):
        self.diamond_arr = SimpleMeshData.diamond_table
        self.c2v = SimpleMeshData.c2v_table
        self.e2c = SimpleMeshData.e2c_table
        self.e2v = SimpleMeshData.e2v_table
        self.c2e = SimpleMeshData.c2e_table
        self.c2e2cO = SimpleMeshData.c2e2cO_table
        self.c2e2c = SimpleMeshData.c2e2c_table
        self.e2c2eO = SimpleMeshData.e2c2eO_table
        self.e2c2e = SimpleMeshData.e2c2e_table
        self.e2c2v = SimpleMeshData.e2c2v_table
        self.v2c = SimpleMeshData.v2c_table
        self.v2e = SimpleMeshData.v2e_table
        self.c2e2c2e2c = SimpleMeshData.c2e2c2e2c_table
        self.n_e2c = self.e2c.shape[1]
        self.n_e2v = self.e2v.shape[1]
        self.n_c2e = self.c2e.shape[1]
        self.n_c2e2cO = self.c2e2cO.shape[1]
        self.n_c2e2c = self.c2e2c.shape[1]
        self.n_e2c2eO = self.e2c2eO.shape[1]
        self.n_e2c2e = self.e2c2e.shape[1]
        self.n_e2c2v = self.e2c2v.shape[1]
        self.n_v2c = self.v2c.shape[1]
        self.n_c2v = self.c2v.shape[1]
        self.n_v2e = self.v2e.shape[1]
        self.n_cells = self.c2e.shape[0]
        self.n_c2e2c2e2c = self.c2e2c2e2c.shape[1]
        self.n_edges = 27
        self.n_vertices = 9
        self.k_level = k_level
        self.size = {
            CellDim: self.n_cells,
            EdgeDim: self.n_edges,
            E2VDim: self.n_e2v,
            E2CDim: self.n_e2c,
            C2EDim: self.n_c2e,
            C2E2CODim: self.n_c2e2cO,
            C2E2CDim: self.n_c2e2c,
            E2C2EODim: self.n_e2c2eO,
            E2C2EDim: self.n_e2c2e,
            V2CDim: self.n_v2c,
            KDim: self.k_level,
            VertexDim: self.n_vertices,
            V2EDim: self.n_v2e,
            CEDim: self.n_cells * self.n_c2e,
            ECDim: self.n_edges * self.n_e2c,
            E2C2VDim: self.n_e2c2v,
            ECVDim: self.n_edges * self.n_e2c2v,
            C2E2C2E2CDim: self.n_c2e2c2e2c,
        }

    def get_c2v_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.c2v, VertexDim, CellDim, self.n_c2v)

    def get_c2e_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.c2e, CellDim, EdgeDim, self.n_c2e)

    def get_c2e2cO_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.c2e2cO, CellDim, CellDim, self.n_c2e2cO)

    def get_c2e2c_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.c2e2c, CellDim, CellDim, self.n_c2e2c)

    def get_e2c2eO_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.e2c2eO, EdgeDim, EdgeDim, self.n_e2c2eO)

    def get_e2c2e_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.e2c2e, EdgeDim, EdgeDim, self.n_e2c2e)

    def get_v2c_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.v2c, VertexDim, CellDim, self.n_v2c)

    def get_v2e_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.v2e, VertexDim, EdgeDim, self.n_v2e)

    def get_e2c_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.e2c, EdgeDim, CellDim, self.n_e2c)

    def get_e2v_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.e2v, EdgeDim, VertexDim, self.n_e2v)

    def get_e2c2v_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.e2c2v, EdgeDim, VertexDim, self.n_e2c2v)

    def get_c2e2c2e2c_offset_provider(self) -> NeighborTableOffsetProvider:
        return NeighborTableOffsetProvider(self.c2e2c2e2c, CellDim, CellDim, self.n_c2e2c2e2c)

    def get_e2ecv_offset_provider(self):
        old_shape = self.e2c2v.shape
        e2ecv_table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(e2ecv_table, EdgeDim, ECVDim, e2ecv_table.shape[1])

    def get_c2ce_offset_provider(self):
        old_shape = self.c2e.shape
        c2ce_table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(c2ce_table, CellDim, CEDim, c2ce_table.shape[1])

    def get_offset_provider(self):
        return {
            "C2E": self.get_c2e_offset_provider(),
            "C2E2CO": self.get_c2e2cO_offset_provider(),
            "C2E2C": self.get_c2e2c_offset_provider(),
            "E2C2EO": self.get_e2c2eO_offset_provider(),
            "E2C2E": self.get_e2c2e_offset_provider(),
            "V2C": self.get_v2c_offset_provider(),
            "V2E": self.get_v2e_offset_provider(),
            "E2C": self.get_e2c_offset_provider(),
            "E2V": self.get_e2v_offset_provider(),
            "E2C2V": self.get_e2c2v_offset_provider(),
            "C2CE": self.get_c2ce_offset_provider(),
            "C2CEC": StridedNeighborOffsetProvider(CellDim, CECDim, self.n_c2e2c),
            "Koff": KDim,
            "C2E2C2E2C": self.get_c2e2c2e2c_offset_provider(),
            "E2ECV": StridedNeighborOffsetProvider(EdgeDim, ECVDim, self.n_e2c2v),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, self.n_e2c),
            "C2CEC": StridedNeighborOffsetProvider(CellDim, CECDim, self.n_c2e2c),
        }
