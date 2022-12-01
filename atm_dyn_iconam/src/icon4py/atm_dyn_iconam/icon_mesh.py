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

import numpy as np
from functional.common import Field
from functional.iterator.embedded import np_as_located_field

from icon4py.atm_dyn_iconam.horizontal import HorizontalMeshConfig
from icon4py.common.dimension import EdgeDim, KDim






class MeshConfig:
    def __init__(self, horizontalMesh: HorizontalMeshConfig):
        self.num_k_levels = 65
        self.n_shift_total = 0
        self.horizontal = horizontalMesh

    def get_num_k_levels(self):
        return self.num_k_levels

    def get_n_shift(self):
        return self.n_shift_total

    def get_num_vertices(self):
        return self.horizontal._num_vertices

    def get_num_edges(self):
        return self.horizontal._num_edges

    def get_num_cells(self):
        return self.horizontal._num_cells



class IconMesh:
    def __init__(self, config: MeshConfig):
        self.config = config
        # TODO [ml] calculate the content of this stuff? read in?
        self._tangent_orientation = np_as_located_field(EdgeDim, KDim)(
            np.zeros(self.config.get_num_edges(), self.config.get_num_k_levels())
        )
        self._triangle_edge_inverse_length = np_as_located_field(EdgeDim)(
            np.zeros(self.config.get_num_edges())
        )
        # normal to triangle edge projected to the location of the vertices
        self._primal_normal_vert = (
            np_as_located_field(EdgeDim)(np.zeros(self.config.get_num_edges())),
            np_as_located_field(EdgeDim)(np.zeros(self.config.get_num_edges())),
        )

        # tangent to triangle edge, projected to the location of the vertices
        self._dual_normal_vert_x = (
            np_as_located_field(EdgeDim)(np.zeros(self.config.get_num_edges())),
            np_as_located_field(EdgeDim)(np.zeros(self.config.get_num_edges())),
        )

    # TODO [ml] geometry
    def tangent_orientation(self):
        return self._tangent_orientation

    def inv_primal_edge_length(self):
        return self._triangle_edge_inverse_length

    def primal_normal_vert(self):
        return self._primal_normal_vert

    def dual_normal_vert_x(self):
        return self._dual_normal_vert_x


class VerticalModelParams:
    def __init__(self, vct_a: np.ndarray, rayleigh_damping_height: float = 12500.0):
        """
        Contains physical parameters defined on the grid.

        Args:
            vct_a: TODO read from vertical_coord_tables!
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


