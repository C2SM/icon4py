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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_59 import (
    mo_solve_nonhydro_stencil_59,
)
from icon4py.common.dimension import CellDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field


def mo_solve_nonhydro_stencil_59_numpy(exner: np.array) -> np.array:
    exner_dyn_incr = exner
    return exner_dyn_incr


def test_mo_solve_nonhydro_stencil_59():
    mesh = SimpleMesh()

    exner = random_field(mesh, CellDim, KDim)
    exner_dyn_incr = zero_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_59_numpy(
        np.asarray(exner),
    )

    mo_solve_nonhydro_stencil_59(
        exner,
        exner_dyn_incr,
        offset_provider={},
    )
    assert np.allclose(exner_dyn_incr, ref)
