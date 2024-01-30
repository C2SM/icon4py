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

from icon4py.model.common.dimension import (
    C2E2C2E2CDim,
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    C2VDim,
    CECDim,
    CECECDim,
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
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.base import BaseGrid
from icon4py.model.common.utils import builder


class IconGrid(BaseGrid):
    def __init__(self):
        """Instantiate a grid according to the ICON model."""
        super().__init__()
        self.start_indices = {}
        self.end_indices = {}
        self.offset_provider_mapping = {
            "C2E2CO": (self._get_offset_provider, C2E2CODim, CellDim, CellDim, True),
            "E2C2V": (self._get_offset_provider, E2C2VDim, EdgeDim, VertexDim, True),
            "V2E": (self._get_offset_provider, V2EDim, VertexDim, EdgeDim, False),
            "V2C": (self._get_offset_provider, V2CDim, VertexDim, CellDim, False),
            "C2E": (self._get_offset_provider, C2EDim, CellDim, EdgeDim, False),
            "E2C": (self._get_offset_provider, E2CDim, EdgeDim, CellDim, True),
            "E2V": (self._get_offset_provider, E2VDim, EdgeDim, VertexDim, True),
            "C2E2C": (self._get_offset_provider, C2E2CDim, CellDim, CellDim, True),
            "E2EC": (self._get_offset_provider_for_sparse_fields, E2CDim, EdgeDim, ECDim),
            "E2ECV": (self._get_offset_provider_for_sparse_fields, E2C2VDim, EdgeDim, ECVDim),
            "C2CEC": (self._get_offset_provider_for_sparse_fields, C2E2CDim, CellDim, CECDim),
            "C2CE": (self._get_offset_provider_for_sparse_fields, C2EDim, CellDim, CEDim),
            "E2C2E": (self._get_offset_provider, E2C2EDim, EdgeDim, EdgeDim, True),
            "E2C2EO": (self._get_offset_provider, E2C2EODim, EdgeDim, EdgeDim, True),
            "Koff": (lambda: KDim,),  # Koff is a special case
            "C2CECEC ": (
                self._get_offset_provider_for_sparse_fields,
                C2E2C2E2CDim,
                CellDim,
                CECECDim,
            ),
        }

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
        return self.config.num_edges if self.config else 0

    @property
    def limited_area(self):
        # defined in mo_grid_nml.f90
        return self.config.limited_area

    @property
    def n_shift(self):
        return self.config.n_shift_total if self.config else 0

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
