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
from icon4py.model.common.dimension import CellDim, KDim

from icon4py.model.atm_dyn_iconam.mo_solve_nonhydro_stencil_50 import (
    mo_solve_nonhydro_stencil_50,
)

from .test_utils.helpers import random_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_50_numpy(
    z_rho_expl: np.array,
    rho_incr: np.array,
    z_exner_expl: np.array,
    exner_incr: np.array,
    iau_wgt_dyn,
) -> tuple[np.array]:
    z_rho_expl = z_rho_expl + iau_wgt_dyn * rho_incr
    z_exner_expl = z_exner_expl + iau_wgt_dyn * exner_incr
    return z_rho_expl, z_exner_expl


def test_mo_solve_nonhydro_stencil_50():
    mesh = SimpleMesh()

    z_exner_expl = random_field(mesh, CellDim, KDim)
    exner_incr = random_field(mesh, CellDim, KDim)
    z_rho_expl = random_field(mesh, CellDim, KDim)
    rho_incr = random_field(mesh, CellDim, KDim)
    iau_wgt_dyn = 8.0

    z_rho_expl_ref, z_exner_expl_ref = mo_solve_nonhydro_stencil_50_numpy(
        np.asarray(z_rho_expl),
        np.asarray(rho_incr),
        np.asarray(z_exner_expl),
        np.asarray(exner_incr),
        iau_wgt_dyn,
    )

    mo_solve_nonhydro_stencil_50(
        z_rho_expl,
        z_exner_expl,
        rho_incr,
        exner_incr,
        iau_wgt_dyn,
        offset_provider={},
    )

    assert np.allclose(z_rho_expl, z_rho_expl_ref)
    assert np.allclose(z_exner_expl, z_exner_expl_ref)
