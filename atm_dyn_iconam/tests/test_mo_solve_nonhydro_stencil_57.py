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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_57 import (
    mo_solve_nonhydro_stencil_57,
)
from icon4py.common.dimension import CellDim, KDim

from .test_utils.helpers import zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_57_numpy(mass_flx_ic: np.array) -> np.array:
    mass_flx_ic = np.zeros_like(mass_flx_ic)
    return mass_flx_ic


def test_mo_solve_nonhydro_stencil_57():
    mesh = SimpleMesh()

    mass_flx_ic = zero_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_57_numpy(np.asarray(mass_flx_ic))
    mo_solve_nonhydro_stencil_57(
        mass_flx_ic,
        offset_provider={},
    )
    assert np.allclose(mass_flx_ic, ref)
