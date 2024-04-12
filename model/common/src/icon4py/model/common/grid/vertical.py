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
import logging
import os
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import KDim
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)

# Choose array backend
if os.environ.get("GT4PY_GPU"):
    import cupy as cp

    xp = cp
else:
    import numpy as np

    xp = np


@dataclass(frozen=True)
class VerticalGridSize:
    num_lev: int


@dataclass(frozen=True)
class VerticalModelParams:
    """
    Contains vertical physical parameters defined on the grid.

    vct_a:  field containing the physical heights of the k level
    rayleigh_damping_height: height [m] of rayleigh damping. Defined `mo_nonhydrostatic_nml.f90` as `damp_height`
    htop_moist_proc: height [m] where moist physics is turned off. Defined in `mo_nonhydrostatic_nml.f90` as `htop_moist_proc`
    """

    vct_a: Field[[KDim], float]
    rayleigh_damping_height: Final[float] = 45000.0
    htop_moist_proc: Final[float] = 22500.0
    index_of_damping_layer: Final[int32] = field(init=False)
    _start_index_for_moist_physics: Final[int32] = field(init=False)
    # TODO: @nfarabullini: check this value # according to mo_init_vgrid.f90 line 329
    nflatlev: Final[int32] = None
    nflat_gradp: Final[int32] = None

    def __post_init__(self):
        vct_a_array = self.vct_a.ndarray
        object.__setattr__(
            self,
            "index_of_damping_layer",
            self._determine_damping_height_index(vct_a_array, self.rayleigh_damping_height),
        )
        object.__setattr__(
            self,
            "_start_index_for_moist_physics",
            self._determine_kstart_moist(vct_a_array, self.htop_moist_proc),
        )
        log.info(f"computation of moist physics start on layer: {self.kstart_moist}")
        log.info(f"end index of Rayleigh damping layer for w: {self.nrdmax} ")

    @property
    def kstart_moist(self):
        """Vertical index for start level of moist physics."""
        return self._start_index_for_moist_physics

    @property
    def nrdmax(self):
        """Vertical index where damping starts."""
        return self.index_of_damping_layer

    @classmethod
    def _determine_kstart_moist(
        cls, vct_a: np.ndarray, top_moist_threshold: float, nshift_total: int = 0
    ) -> int32:
        n_levels = vct_a.shape[0]
        interface_height = 0.5 * (vct_a[: n_levels - 1 - nshift_total] + vct_a[1 + nshift_total :])
        return int32(xp.min(xp.where(interface_height < top_moist_threshold)[0]).item())

    @classmethod
    def _determine_damping_height_index(cls, vct_a: np.ndarray, damping_height: float):
        return int32(xp.argmax(xp.where(vct_a >= damping_height)[0]).item())

    @property
    def physical_heights(self) -> Field[[KDim], float]:
        return self.vct_a


class VerticalGridConfig:
    def __init__(self, num_lev: int):
        self._num_lev = num_lev

    @property
    def num_lev(self) -> int:
        return self._num_lev
