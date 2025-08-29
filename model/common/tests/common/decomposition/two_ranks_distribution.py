# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import numpy as np


TWO_RANKS_DISTRIBUTION: np.ndarray = np.ones(10)
TWO_RANKS_DISTRIBUTION[5, 6, 10] = 0


# TODO define all the rest
CELL_OWN: Final[dict[int, list[int]]] = {
    0: [6, 7, 10],
    1: [0, 1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 16, 17],
}
EDGE_OWN: Final[dict[int, list[int]]] = {
    0: [13, 14],
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
}
VERTEX_OWN: Final[dict[int, list[int]]] = {0: [], 1: [0, 1, 2, 3, 4, 5, 6, 7, 8]}

CELL_FIRST_HALO_LINE = {0: [3, 4, 11, 13, 9], 1: [6, 7, 14]}
CELL_SECOND_HALO_LINE = {0: [0, 1, 5, 8, 14, 17, 16], 1: []}
EDGE_FIRST_HALO_LINE = {0: [9, 12, 17, 21, 10], 1: []}
EDGE_SECOND_HALO_LINE = {0: [1, 2, 5, 4, 15, 16, 24, 25, 26, 22, 23, 18, 11], 1: [14, 13]}
