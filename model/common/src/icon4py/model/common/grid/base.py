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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np
from gt4py.next.common import Dimension

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.utils import builder


@dataclass(
    frozen=True,
)
class GridConfig:
    horizontal_config: HorizontalGridSize
    vertical_config: VerticalGridSize
    limited_area: bool = True
    n_shift_total: int = 0
    lvertnest: bool = False

    @property
    def num_levels(self):
        return self.vertical_config.num_lev

    @property
    def num_vertices(self):
        return self.horizontal_config.num_vertices

    @property
    def num_edges(self):
        return self.horizontal_config.num_edges

    @property
    def num_cells(self):
        return self.horizontal_config.num_cells


class BaseGrid(ABC):
    def __init__(self):
        self.config: GridConfig = None
        self.connectivities: Dict[Dimension, np.ndarray] = {}
        self.size: Dict[Dimension, int] = {}

    @property
    @abstractmethod
    def num_cells(self) -> int:
        pass

    @property
    @abstractmethod
    def num_vertices(self) -> int:
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        pass

    @property
    @abstractmethod
    def num_levels(self) -> int:
        pass

    @abstractmethod
    def get_offset_provider(self) -> dict:
        pass

    @builder
    def with_connectivities(self, connectivity: Dict[Dimension, np.ndarray]):
        self.connectivities.update({d: k.astype(int) for d, k in connectivity.items()})
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    @builder
    def with_config(self, config: GridConfig):
        self.config = config
        self._update_size(config)

    def _update_size(self, config: GridConfig):
        self.size[VertexDim] = config.num_vertices
        self.size[CellDim] = config.num_cells
        self.size[EdgeDim] = config.num_edges
        self.size[KDim] = config.num_levels
