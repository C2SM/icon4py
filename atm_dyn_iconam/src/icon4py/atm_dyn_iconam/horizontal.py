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
from enum import Enum
from typing import Final

from functional.common import Dimension, DimensionKind

@dataclass(frozen=True)
class HorizontalMeshParams:
    NUM_GHOST_ROWS:Final[int] = 2
    GRF_BOUNDARY_WIDTH_CELL:Final[int] = 12
    MIN_RL_CELL_INT:Final[int] = 4
    MIN_RL_CELL:Final[int] = MIN_RL_CELL_INT - 2 * NUM_GHOST_ROWS
    MAX_RL_CELL:Final[int] = 13
    MIN_RL_VERTEX_INT:Final[int] = MIN_RL_CELL_INT
    MIN_RL_VERTEX:Final[int] = MIN_RL_VERTEX_INT - (NUM_GHOST_ROWS + 1)
    MAX_RL_VERTEX:Final[int] = MAX_RL_CELL
    MIN_RL_EDGE_INT:Final[int] = 2 * MIN_RL_CELL_INT
    MIN_RL_EDGE:Final[int] = MIN_RL_EDGE_INT - (2 * NUM_GHOST_ROWS + 1)
    MAX_RL_EDGE:Final[int] = 2 * MAX_RL_CELL
    GRF_BOUNDARY_WIDTH_EDGES: Final[int] = 9

class HorizontalMarkerIndex(Enum):
    START_PROG_CELL = HorizontalMeshParams.GRF_BOUNDARY_WIDTH_CELL + 1
    END_PROG_CELL = HorizontalMeshParams.MIN_RL_CELL_INT

class HorizontalMeshConfig:
    def __init__(self, num_vertices: int, num_edges: int, num_cells: int):
        self._num_vertices = num_vertices
        self._num_edges = num_edges
        self._num_cells = num_cells

    def get_num_vertices(self):
        return self._num_vertices

    def get_num_edges(self):
        return self._num_edges

    def get_num_cells(self):
        return self._num_cells






