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
from enum import Enum

from functional.common import Dimension, DimensionKind


# rather use DSL (dusk) names
class HorizontalIndexMarker(Enum):
    INTERIOR = "0"
    NUDGING = "grf_bdywidth"
    HALO= "min_rl_int"
    END="min_rl"
    LOCAL_BOUNDARY="1"
    MAX_RL="max_rl"

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

    def get_index(self, dim:Dimension, marker:HorizontalIndexMarker) -> int:
        if dim.kind != DimensionKind.HORIZONTAL:
            raise ValueError("only defined for {} dimension kind ", DimensionKind.HORIZONTAL)
        return 0




