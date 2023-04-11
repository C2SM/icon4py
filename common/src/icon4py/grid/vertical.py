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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.common.dimension import KDim


class VerticalGridConfig:
    def __init__(self, num_lev: int):
        self._num_lev = num_lev

    @property
    def num_lev(self) -> int:
        return self._num_lev


class VerticalModelParams:
    def __init__(self, vct_a: Field[[KDim], float], rayleigh_damping_height: float):
        """
        Contains vertical physical parameters defined on the grid.

        Args:
            vct_a:  field containing the physical heights of the k level
            rayleigh_damping_height: height of rayleigh damping in [m] mo_nonhydro_nml
        """
        self._rayleigh_damping_height = rayleigh_damping_height
        self._vct_a = vct_a
        self._index_of_damping_height = int32(
            np.argmax(
                np.where(np.asarray(self._vct_a) >= self._rayleigh_damping_height)
            )
        )

    @property
    def index_of_damping_layer(self) -> int32:
        return self._index_of_damping_height

    @property
    def physical_heights(self) -> Field[[KDim], float]:
        return self._vct_a

    @property
    def rayleigh_damping_height(self) -> float:
        return self._rayleigh_damping_height
