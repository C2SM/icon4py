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

import numpy as np
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider

from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    C2VDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECDim,
    ECVDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.base import BaseGrid
from icon4py.model.common.grid.utils import neighbortable_offset_provider_for_1d_sparse_fields
from icon4py.model.common.utils import builder


class IconGrid(BaseGrid):
    def __init__(self):
        """Instantiate a grid according to the ICON model."""
        super().__init__()
        self.start_indices = {}
        self.end_indices = {}

    @builder
    def with_start_end_indices(
        self, dim: Dimension, start_indices: np.ndarray, end_indices: np.ndarray
    ):
        self.start_indices[dim] = start_indices.astype(int32)
        self.end_indices[dim] = end_indices.astype(int32)

    @property
    def num_levels(self):
        return self.config.num_levels if self.config else 0

    @property
    def num_cells(self):
        return self.config.num_cells if self.config else 0

    @property
    def num_vertices(self):
        return self.config.num_vertices if self.config else 0

    @property
    def num_edges(self):
        return self.config.num_edges

    @property
    def limited_area(self):
        # defined in mo_grid_nml.f90
        return self.config.limited_area

    @property
    def n_shift(self):
        return self.config.n_shift_total if self.config else 0

    @property
    def nflat_gradp(self):
        return (
            self.config.num_levels if self.config else 0
        )  # according to line 1168 in mo_vertical_grid.f90

    @property
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

    def get_c2e_offset_provider(self):
        table = self.connectivities[C2EDim]
        return NeighborTableOffsetProvider(table, CellDim, EdgeDim, table.shape[1])

    def get_e2c_offset_provider(self):
        table = self.connectivities[E2CDim]
        return NeighborTableOffsetProvider(table, EdgeDim, CellDim, table.shape[1])

    def get_e2v_offset_provider(self):
        table = self.connectivities[E2VDim]
        return NeighborTableOffsetProvider(table, EdgeDim, VertexDim, table.shape[1])

    def get_c2e2c_offset_provider(self):
        table = self.connectivities[C2E2CDim]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])

    def get_e2ec_offset_provider(self):
        old_shape = self.connectivities[E2CDim].shape
        e2ec_table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(e2ec_table, EdgeDim, ECDim, e2ec_table.shape[1])

    def get_c2e2co_offset_provider(self):
        table = self.connectivities[C2E2CODim]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])

    def get_e2c2v_offset_provider(self):
        table = self.connectivities[E2C2VDim]
        return NeighborTableOffsetProvider(table, EdgeDim, VertexDim, table.shape[1])

    def get_v2e_offset_provider(self):
        table = self.connectivities[V2EDim]
        return NeighborTableOffsetProvider(table, VertexDim, EdgeDim, table.shape[1])

    def get_v2c_offset_provider(self):
        table = self.connectivities[V2CDim]
        return NeighborTableOffsetProvider(table, VertexDim, CellDim, table.shape[1])

    def get_c2v_offset_provider(self):
        table = self.connectivities[C2VDim]
        return NeighborTableOffsetProvider(table, VertexDim, CellDim, table.shape[1])

    def get_e2ecv_offset_provider(self):
        return neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities[E2C2VDim].shape, EdgeDim, ECVDim
        )

    def get_c2cec_offset_provider(self):
        return neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities[C2E2CDim].shape, CellDim, CECDim
        )

    def get_c2ce_offset_provider(self):
        return neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities[C2EDim].shape, CellDim, CEDim
        )

    def get_e2c2e_offset_provider(self):
        table = self.connectivities[E2C2EDim]
        return NeighborTableOffsetProvider(table, EdgeDim, EdgeDim, table.shape[1])

    def get_e2c2eo_offset_provider(self):
        table = self.connectivities[E2C2EODim]
        return NeighborTableOffsetProvider(table, EdgeDim, EdgeDim, table.shape[1])

    def get_offset_provider(self):
        return {
            "C2E": self.get_c2e_offset_provider(),
            "E2C": self.get_e2c_offset_provider(),
            "E2V": self.get_e2v_offset_provider(),
            "C2E2C": self.get_c2e2c_offset_provider(),
            "E2EC": self.get_e2ec_offset_provider(),
            "C2E2CO": self.get_c2e2co_offset_provider(),
            "E2C2V": self.get_e2c2v_offset_provider(),
            "V2E": self.get_v2e_offset_provider(),
            "V2C": self.get_v2c_offset_provider(),
            # "C2V": self.get_c2v_offset_provider(),
            "E2ECV": self.get_e2ecv_offset_provider(),
            "C2CEC": self.get_c2cec_offset_provider(),
            "C2CE": self.get_c2ce_offset_provider(),
            "E2C2E": self.get_e2c2e_offset_provider(),
            "E2C2EO": self.get_e2c2eo_offset_provider(),
        }
