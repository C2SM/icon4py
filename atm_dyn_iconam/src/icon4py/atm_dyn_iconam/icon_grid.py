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
from typing import Tuple

import numpy as np
from functional.common import Field, Dimension, DimensionKind
from functional.iterator.embedded import np_as_located_field, NeighborTableOffsetProvider

from icon4py.atm_dyn_iconam.horizontal import HorizontalMeshConfig, HorizontalMarkerIndex
from icon4py.common.dimension import EdgeDim, KDim, CellDim, VertexDim


class MeshConfig:
    def __init__(self, horizontalMesh: HorizontalMeshConfig):
        self._num_k_levels = 65
        self._n_shift_total = 0
        self._horizontal = horizontalMesh

    @property
    def num_k_levels(self):
        return self._num_k_levels

    @property
    def n_shift_total(self):
        return self._n_shift_total

    @property
    def get_num_vertices(self):
        return self._horizontal._num_vertices

    def get_num_edges(self):
        return self._horizontal._num_edges

    def get_num_cells(self):
        return self._horizontal._num_cells

def builder(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper

class IconGrid:
    def __init__(self):
        self.config:MeshConfig = None
        self.start_indices = {}
        self.end_indices = {}
        self.connectivities={}

    @builder
    def with_config(self, config: MeshConfig):
        self.config = config

    @builder
    def with_start_end_indices(self, dim:Dimension, start_indices:np.ndarray, end_indices: np.ndarray):
        self.start_indices[dim] = start_indices
        self.end_indices[dim] = end_indices

    @builder
    def with_connectivity(self, **connectivity):
        self.connectivities.update(**connectivity)

    def k_levels(self):
        return self.config.num_k_levels



    def get_indices_from_to(self, dim:Dimension, start_marker:int, end_marker: int) -> Tuple[int, int]:
        if dim.kind != DimensionKind.HORIZONTAL:
            raise ValueError("only defined for {} dimension kind ", DimensionKind.HORIZONTAL)
        #print(self.start_indices[dim])
        #print(self.end_indices[dim])
        return self.start_indices[dim][start_marker], self.end_indices[dim][end_marker]

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

    def get_c2e2c0_connectivity(self):
        table = self.connectivities["c2e2c0"]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])


class VerticalModelParams:
    def __init__(self, vct_a: np.ndarray, rayleigh_damping_height: float = 12500.0):
        """
        Contains physical parameters defined on the grid.

        Args:
            vct_a:
            rayleigh_damping_height height of rayleigh damping in [m] mo_nonhydro_nml
        """
        self.rayleigh_damping_height = rayleigh_damping_height
        self.vct_a = vct_a
        # TODO klevels in ICON are inverted!
        self.index_of_damping_height = np.argmax(
            self.vct_a >= self.rayleigh_damping_height
        )

    def get_index_of_damping_layer(self):
        return self.index_of_damping_height

    def get_physical_heights(self) -> Field[[KDim], float]:
        return np_as_located_field(KDim)(self.vct_a)

    def init_nabla2_factor_in_upper_damping_zone(
        self, k_size: int
    ) -> Field[[KDim], float]:
        # this assumes n_shift == 0
        buffer = np.zeros(k_size)
        buffer[2 : self.index_of_damping_height] = (
            1.0
            / 12.0
            * (
                self.vct_a[2 : self.index_of_damping_height]
                - self.vct_a[self.index_of_damping_height + 1]
            )
            / (self.vct_a[2] - self.vct_a[self.index_of_damping_height + 1])
        )
        return np_as_located_field(KDim)(buffer)


