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

from dataclasses import Field, dataclass, field
from typing import Final

import numpy as np
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next import common
from icon4py.model.common.dimension import KDim


@dataclass(frozen=True)
class VerticalGridSize:
    num_lev: int


@dataclass(frozen=True)
class VerticalModelParams:
    """
    Contains vertical physical parameters defined on the grid.

    vct_a:  field containing the physical heights of the k level
    rayleigh_damping_height: height of rayleigh damping in [m] mo_nonhydro_nml
    """

    vct_a: common.Field
    rayleigh_damping_height: Final[float]
    index_of_damping_layer: Final[int32] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "index_of_damping_layer",
            self._determine_damping_height_index(
                np.asarray(self.vct_a), self.rayleigh_damping_height
            ),
        )

    @classmethod
    def _determine_damping_height_index(cls, vct_a: np.ndarray, damping_height: float):
        return int32(np.argmax(np.where(vct_a >= damping_height)))

    @property
    def physical_heights(self) -> Field[[KDim], float]:
        return self.vct_a
