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
import dataclasses
import uuid

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.base import BaseGrid, GridConfig, HorizontalGridSize

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
from icon4py.model.common.grid.vertical import VerticalGridConfig


@dataclasses.dataclass
class SimpleGridData:
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
        ],
        dtype=gtx.int32,
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
            [2, 3, 0, 5],  # 7
            [2, 5, 1, 3],  # 8
            [3, 4, 0, 7],  # 9
            [3, 7, 4, 6],  # 10
            [3, 6, 5, 7],  # 11
            [4, 5, 1, 8],  # 12
            [4, 8, 5, 7],  # 13
            [4, 7, 3, 8],  # 14
            [5, 3, 2, 6],  # 15
            [5, 6, 3, 8],  # 16
            [5, 8, 4, 6],  # 17
            [6, 7, 3, 1],  # 18
            [6, 1, 7, 0],  # 19
            [6, 0, 1, 8],  # 20
            [7, 8, 4, 2],  # 21
            [7, 2, 8, 1],  # 22
            [7, 1, 6, 2],  # 23
            [8, 6, 5, 0],  # 24
            [8, 0, 6, 2],  # 25
            [8, 2, 7, 0],  # 26
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
    )

    c2e2c2e_table = np.asarray(
        [
            [19, 20, 0, 1, 2, 9, 4, 5, 12],  # 0c
            [22, 23, 3, 4, 5, 12, 7, 8, 15],
            [25, 26, 6, 7, 8, 15, 1, 2, 9],
            [0, 1, 5, 6, 7, 2, 9, 10, 14],
            [3, 4, 8, 0, 1, 5, 12, 13, 17],  # 4c
            [6, 7, 2, 3, 4, 8, 15, 16, 11],
            [1, 2, 9, 10, 11, 18, 13, 14, 21],
            [4, 5, 12, 13, 14, 21, 16, 17, 24],
            [7, 8, 15, 16, 17, 24, 10, 11, 18],
            [9, 10, 14, 15, 16, 11, 18, 19, 23],  # 9c
            [12, 13, 17, 9, 10, 14, 21, 22, 26],
            [15, 16, 11, 12, 13, 17, 24, 25, 20],
            [10, 11, 18, 19, 20, 0, 22, 23, 3],
            [13, 14, 21, 22, 23, 3, 25, 26, 6],
            [16, 17, 24, 25, 26, 6, 19, 20, 0],  # 14c
            [18, 19, 23, 24, 25, 20, 0, 1, 5],
            [21, 22, 26, 18, 19, 23, 3, 4, 8],
            [24, 25, 20, 21, 22, 26, 6, 7, 2],  # 17c
        ],
        dtype=gtx.int32,
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
        ],
        dtype=gtx.int32,
    )


class SimpleGrid(BaseGrid):
    _CELLS = 18
    _EDGES = 27
    _VERTICES = 9

    def __init__(self):
        """Instantiate a SimpleGrid used for testing purposes."""
        super().__init__()

        self._configure()
        self.offset_provider_mapping = {
            "C2E": (self._get_offset_provider, dims.C2EDim, dims.CellDim, dims.EdgeDim),
            "C2E2CO": (self._get_offset_provider, dims.C2E2CODim, dims.CellDim, dims.CellDim),
            "C2E2C": (self._get_offset_provider, dims.C2E2CDim, dims.CellDim, dims.CellDim),
            "E2C2EO": (self._get_offset_provider, dims.E2C2EODim, dims.EdgeDim, dims.EdgeDim),
            "E2C2E": (self._get_offset_provider, dims.E2C2EDim, dims.EdgeDim, dims.EdgeDim),
            "V2C": (self._get_offset_provider, dims.V2CDim, dims.VertexDim, dims.CellDim),
            "V2E": (self._get_offset_provider, dims.V2EDim, dims.VertexDim, dims.EdgeDim),
            "E2C": (self._get_offset_provider, dims.E2CDim, dims.EdgeDim, dims.CellDim),
            "E2V": (self._get_offset_provider, dims.E2VDim, dims.EdgeDim, dims.VertexDim),
            "E2C2V": (self._get_offset_provider, dims.E2C2VDim, dims.EdgeDim, dims.VertexDim),
            "C2CE": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2EDim,
                dims.CellDim,
                dims.CEDim,
            ),
            "Koff": (lambda: dims.KDim,),  # Koff is a special case
            "C2E2C2E": (self._get_offset_provider, dims.C2E2C2EDim, dims.CellDim, dims.EdgeDim),
            "C2E2C2E2C": (self._get_offset_provider, dims.C2E2C2E2CDim, dims.CellDim, dims.CellDim),
            "E2ECV": (
                self._get_offset_provider_for_sparse_fields,
                dims.E2C2VDim,
                dims.EdgeDim,
                dims.ECVDim,
            ),
            "E2EC": (
                self._get_offset_provider_for_sparse_fields,
                dims.E2CDim,
                dims.EdgeDim,
                dims.ECDim,
            ),
            "C2CEC": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2E2CDim,
                dims.CellDim,
                dims.CECDim,
            ),
            "C2CECEC": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2E2C2E2CDim,
                dims.CellDim,
                dims.CECECDim,
            ),
        }

    @property
    def num_cells(self) -> int:
        return self.config.num_cells

    @property
    def num_vertices(self) -> int:
        return self.config.num_vertices

    @property
    def num_edges(self) -> int:
        return self.config.num_edges

    @property
    def diamond_table(self) -> np.ndarray:
        return SimpleGridData.e2c2v_table

    @property
    def num_levels(self) -> int:
        return self.config.num_levels

    @property
    def id(self) -> uuid.UUID:
        return uuid.UUID("bd68594d-e151-459c-9fdc-32e989d3ca85")

    def _has_skip_values(self, dimension: gtx.Dimension) -> bool:
        return False

    def _configure(self):
        horizontal_grid_size = HorizontalGridSize(
            num_vertices=self._VERTICES, num_edges=self._EDGES, num_cells=self._CELLS
        )
        vertical_grid_config = VerticalGridConfig(num_levels=10)
        config = GridConfig(
            horizontal_config=horizontal_grid_size,
            vertical_size=vertical_grid_config.num_levels,
        )

        connectivity_dict = {
            dims.C2VDim: SimpleGridData.c2v_table,
            dims.E2CDim: SimpleGridData.e2c_table,
            dims.E2VDim: SimpleGridData.e2v_table,
            dims.C2EDim: SimpleGridData.c2e_table,
            dims.C2E2CODim: SimpleGridData.c2e2cO_table,
            dims.C2E2CDim: SimpleGridData.c2e2c_table,
            dims.E2C2EODim: SimpleGridData.e2c2eO_table,
            dims.E2C2EDim: SimpleGridData.e2c2e_table,
            dims.E2C2VDim: SimpleGridData.e2c2v_table,
            dims.V2CDim: SimpleGridData.v2c_table,
            dims.V2EDim: SimpleGridData.v2e_table,
            dims.C2E2C2EDim: SimpleGridData.c2e2c2e_table,
            dims.C2E2C2E2CDim: SimpleGridData.c2e2c2e2c_table,
        }

        self.with_config(config).with_connectivities(connectivity_dict)
        self.update_size_connectivities(
            {
                dims.ECVDim: self.size[dims.EdgeDim] * self.size[dims.E2C2VDim],
                dims.CEDim: self.size[dims.CellDim] * self.size[dims.C2EDim],
                dims.ECDim: self.size[dims.EdgeDim] * self.size[dims.E2CDim],
            }
        )
