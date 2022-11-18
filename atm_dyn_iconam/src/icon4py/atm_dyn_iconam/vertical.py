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
from functional.common import Field
from functional.iterator.embedded import np_as_located_field

from icon4py.common.dimension import KDim


class VerticalGridConfig:
    def __init__(self):
        self.num_k_levels = 65
        self.n_shift_total = 0

    def get_k_size(self):
        return self.num_k_levels

    def get_n_shift(self):
        return self.n_shift_total


class VerticalModelParams:
    def __init__(self, vct_a: np.ndarray, rayleigh_damping_height: float = 12500.0):
        """
        Arguments.

        - vct_a: # read from vertical_coord_tables
        - rayleigh_damping_height height of rayleigh damping in [m] mo_nonhydro_nml
        """
        self.rayleigh_damping_height = rayleigh_damping_height
        self.vct_a = vct_a
        # TODO klevels in ICON are inversed
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
