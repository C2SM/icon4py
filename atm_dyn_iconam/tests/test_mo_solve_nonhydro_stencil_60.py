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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_60 import (
    mo_solve_nonhydro_stencil_60,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_60_numpy(
    exner: np.array,
    ddt_exner_phy: np.array,
    exner_dyn_incr: np.array,
    ndyn_substeps_var: float,
    dtime: float,
) -> np.array:
    exner_dyn_incr = exner - (
        exner_dyn_incr + ndyn_substeps_var * dtime * ddt_exner_phy
    )
    return exner_dyn_incr


def test_mo_solve_nonhydro_stencil_60():
    mesh = SimpleMesh()

    ndyn_substeps_var, dtime = float(10.0), float(12.0)
    exner = random_field(mesh, CellDim, KDim)
    ddt_exner_phy = random_field(mesh, CellDim, KDim)
    exner_dyn_incr = random_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_60_numpy(
        np.asarray(exner),
        np.asarray(ddt_exner_phy),
        np.asarray(exner_dyn_incr),
        ndyn_substeps_var,
        dtime,
    )

    mo_solve_nonhydro_stencil_60(
        exner,
        ddt_exner_phy,
        exner_dyn_incr,
        ndyn_substeps_var,
        dtime,
        offset_provider={},
    )
    assert np.allclose(exner_dyn_incr, ref)
