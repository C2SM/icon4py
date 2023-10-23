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
    VertexDim,
)
from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.mesh import BaseMesh
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
    def k_levels(self):
        return self.vertical_config.num_lev

    @property
    def n_vertices(self):
        return self.horizontal_config.n_vertices

    @property
    def n_edges(self):
        return self.horizontal_config.n_edges

    @property
    def n_cells(self):
        return self.horizontal_config.n_cells


class IconGrid(BaseMesh):
    def __init__(self):
        super().__init__()
        self.start_indices = {}
        self.end_indices = {}

    @builder
    def with_start_end_indices(
        self, dim: Dimension, start_indices: np.ndarray, end_indices: np.ndarray
    ):
        self.start_indices[dim] = start_indices.astype(int32)
        self.end_indices[dim] = end_indices.astype(int32)

    def limited_area(self):
        # defined in mo_grid_nml.f90
        return self.config.limited_area

    def n_shift(self):
        return self.config.n_shift_total if self.config else 0

    def n_lev(self):
        return self.config.k_levels if self.config else 0

    def nflat_gradp(self):
        return (
            self.config.k_levels if self.config else 0
        )  # according to line 1168 in mo_vertical_grid.f90

    def n_cells(self):
        return self.config.n_cells if self.config else 0

    def n_vertices(self):
        return self.config.n_vertices if self.config else 0

    def n_edges(self):
        return self.config.n_edges

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

    def get_offset_provider(self):
        return {
            "C2E": self.get_c2e_connectivity(),
            "E2C": self.get_e2c_connectivity(),
            "E2V": self.get_e2v_connectivity(),
            "C2E2C": self.get_c2e2c_connectivity(),
            "E2EC": self.get_e2ec_connectivity(),
            "C2E2CO": self.get_c2e2co_connectivity(),
            "E2C2V": self.get_e2c2v_connectivity(),
            "V2E": self.get_v2e_connectivity(),
            "V2C": self.get_v2c_connectivity(),
            "C2V": self.get_c2v_connectivity(),
            "E2ECV": self.get_e2ecv_connectivity(),
            "C2CEC": self.get_c2cec_connectivity(),
            "C2CE": self.get_c2ce_connectivity(),
            "E2C2E": self.get_e2c2e_connectivity(),
            "E2C2EO": self.get_e2c2eo_connectivity(),
        }
