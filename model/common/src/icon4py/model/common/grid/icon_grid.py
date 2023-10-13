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
from typing import Dict

import numpy as np
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider

from icon4py.model.common.dimension import (
    CECDim,
    CEDim,
    CellDim,
    ECDim,
    ECVDim,
    EdgeDim,
    KDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.utils import builder


class VerticalMeshConfig:
    def __init__(self, num_lev: int):
        self._num_lev = num_lev

    @property
    def num_lev(self) -> int:
        return self._num_lev


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
    def num_k_levels(self):
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
        self.start_indices[dim] = start_indices.astype(int32)
        self.end_indices[dim] = end_indices.astype(int32)

    @builder
    def with_connectivities(self, connectivity: Dict[Dimension, np.ndarray]):
        self.connectivities.update(
            {d.value.lower(): k.astype(int) for d, k in connectivity.items()}
        )
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    def limited_area(self):
        # defined in mo_grid_nml.f90
        return self.config.limited_area

    def n_shift(self):
        return self.config.n_shift_total if self.config else 0

    def n_lev(self):
        return self.config.num_k_levels if self.config else 0

    def nflat_gradp(self):
        return (
            self.config.num_k_levels if self.config else 0
        )  # according to line 1168 in mo_vertical_grid.f90

    def num_cells(self):
        return self.config.num_cells if self.config else 0

    def num_vertices(self):
        return self.config.num_vertices if self.config else 0

    def num_edges(self):
        return self.config.num_edges

    def lvert_nest(self):
        return True if self.config.lvertnest else False

    def get_start_index(self, dim: Dimension, marker: int) -> int32:
        """
        Use to specify lower end of domains of a field for field_operators.

        For a given dimension, returns the start index of the
        horizontal region in a field given by the marker.
        """
        return self.start_indices[dim][marker]

    def get_end_index(self, dim: Dimension, marker: int) -> int32:
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

    def get_e2ec_connectivity(self):
        old_shape = self.connectivities["e2c"].shape
        e2ec_table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(e2ec_table, EdgeDim, ECDim, e2ec_table.shape[1])

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
        return self._neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities["e2c2v"].shape, EdgeDim, ECVDim
        )

    def _neighbortable_offset_provider_for_1d_sparse_fields(
        self,
        old_shape: tuple[int, int],
        origin_axis: Dimension,
        neighbor_axis: Dimension,
    ):
        table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(table, origin_axis, neighbor_axis, table.shape[1])

    def get_c2cec_connectivity(self):
        return self._neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities["c2e2c"].shape, CellDim, CECDim
        )

    def get_c2ce_connectivity(self):
        return self._neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities["c2e"].shape, CellDim, CEDim
        )

    def get_e2c2e_connectivity(self):
        table = self.connectivities["e2c2e"]
        return NeighborTableOffsetProvider(table, EdgeDim, EdgeDim, table.shape[1])

    def get_e2c2eo_connectivity(self):
        table = self.connectivities["e2c2eo"]
        return NeighborTableOffsetProvider(table, EdgeDim, EdgeDim, table.shape[1])
