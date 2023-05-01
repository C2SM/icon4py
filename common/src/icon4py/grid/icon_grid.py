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
from typing import Dict

import numpy as np
from gt4py.next.common import Dimension
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider

from icon4py.common.dimension import CellDim, ECVDim, EdgeDim, KDim, VertexDim
from icon4py.grid.horizontal import HorizontalGridSize
from icon4py.grid.vertical import VerticalGridConfig


class GridConfig:
    def __init__(
        self,
        horizontal_config: HorizontalGridSize,
        vertical_config: VerticalGridConfig,
        limited_area=True,
    ):
        self._vertical = vertical_config
        self._n_shift_total = 0
        self._limited_area = limited_area
        self._horizontal = horizontal_config

    @property
    def limited_area(self):
        return self._limited_area

    @property
    def num_k_levels(self):
        return self._vertical.num_lev

    @property
    def n_shift_total(self):
        return self._n_shift_total

    @property
    def num_vertices(self):
        return self._horizontal.num_vertices

    @property
    def num_edges(self):
        return self._horizontal.num_edges

    @property
    def num_cells(self):
        return self._horizontal.num_cells


def builder(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return wrapper


class IconGrid:
    def __init__(self):
        self.config: GridConfig = None

        self.start_indices = {}
        self.end_indices = {}
        self.connectivities: Dict[str, np.ndarray] = {}
        self.size: Dict[Dimension, int] = {}

    def _update_size(self, config: GridConfig):
        self.size[VertexDim] = config.num_vertices
        self.size[CellDim] = config.num_cells
        self.size[EdgeDim] = config.num_edges
        self.size[KDim] = config.num_k_levels

    @builder
    def with_config(self, config: GridConfig):
        self.config = config
        self._update_size(config)

    @builder
    def with_start_end_indices(
        self, dim: Dimension, start_indices: np.ndarray, end_indices: np.ndarray
    ):
        self.start_indices[dim] = start_indices.astype(int)
        self.end_indices[dim] = end_indices.astype(int)

    @builder
    def with_connectivities(self, connectivity: Dict[Dimension, np.ndarray]):
        self.connectivities.update(
            {d.value.lower(): k.astype(int) for d, k in connectivity.items()}
        )
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    def limited_area(self):
        # TODO defined in mo_grid_nml.f90
        return self.config.limited_area

    def n_lev(self):
        return self.config.num_k_levels if self.config else 0

    def num_cells(self):
        return self.config.num_cells if self.config else 0

    def num_vertices(self):
        return self.config.num_vertices if self.config else 0

    def num_edges(self):
        return self.config.num_edges

    def get_start_index(self, dim: Dimension, marker: int):
        """
        Use to specify lower end of domains of a field for field_operators.

        For a given dimension, returns the start index of the
        horizontal region in a field given by the marker.
        """
        return self.start_indices[dim][marker]

    def get_end_index(self, dim: Dimension, marker: int):
        """
        Use to specify upper end of domains of a field for field_operators.

        For a given dimension, returns the end index of the
        horizontal region in a field given by the marker.
        """
        return self.end_indices[dim][marker]

    def get_c2e_connectivity(self):
        table = self.connectivities["c2e"]
        return NeighborTableOffsetProvider(table, CellDim, EdgeDim, table.shape[1])

    def get_e2c_connectivity(self):
        table = self.connectivities["e2c"]
        return NeighborTableOffsetProvider(table, EdgeDim, CellDim, table.shape[1])

    def get_e2v_connectivity(self):
        table = self.connectivities["e2v"]
        return NeighborTableOffsetProvider(table, EdgeDim, VertexDim, table.shape[1])

    def get_c2e2c_connectivity(self):
        table = self.connectivities["c2e2c"]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])

    def get_c2e2co_connectivity(self):
        table = self.connectivities["c2e2co"]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])

    def get_e2c2v_connectivity(self):
        table = self.connectivities["e2c2v"]
        return NeighborTableOffsetProvider(table, EdgeDim, VertexDim, table.shape[1])

    def get_v2e_connectivity(self):
        table = self.connectivities["v2e"]
        return NeighborTableOffsetProvider(table, VertexDim, EdgeDim, table.shape[1])

    def get_v2c_connectivity(self):
        table = self.connectivities["v2c"]
        return NeighborTableOffsetProvider(table, VertexDim, CellDim, table.shape[1])

    def get_c2v_connectivity(self):
        table = self.connectivities["c2v"]
        return NeighborTableOffsetProvider(table, VertexDim, CellDim, table.shape[1])

    def get_e2ecv_connectivity(self):
        old_shape = self.connectivities["e2c2v"].shape
        v2ecv_table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(
            v2ecv_table, EdgeDim, ECVDim, v2ecv_table.shape[1]
        )
