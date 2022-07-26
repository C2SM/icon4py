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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_02 import (
    mo_solve_nonhydro_stencil_02,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_02_z_exner_ex_pr_numpy(
    exner: np.array, exner_ref_mc: np.array, exner_pr: np.array, exner_exfac: np.array
) -> np.array:
    z_exner_ex_pr = (1 + exner_exfac) * (exner - exner_ref_mc) - exner_exfac * exner_pr
    return z_exner_ex_pr


def mo_solve_nonhydro_stencil_02_exner_pr_numpy(
    exner: np.array, exner_ref_mc: np.array
) -> np.array:
    exner_pr = exner - exner_ref_mc
    return exner_pr


def mo_solve_nonhydro_stencil_02_numpy(
    exner: np.array, exner_ref_mc: np.array, exner_pr: np.array, exner_exfac: np.array
) -> tuple[np.array]:
    z_exner_pr = mo_solve_nonhydro_stencil_02_z_exner_ex_pr_numpy(
        exner, exner_ref_mc, exner_pr, exner_exfac
    )
    exner_pr = mo_solve_nonhydro_stencil_02_exner_pr_numpy(exner, exner_ref_mc)
    return z_exner_pr, exner_pr


def test_mo_solve_nonhydro_stencil_02():
    mesh = SimpleMesh()

    exner = random_field(mesh, CellDim, KDim)
    exner_ref_mc = random_field(mesh, CellDim, KDim)
    exner_pr = zero_field(mesh, CellDim, KDim)
    exner_exfac = random_field(mesh, CellDim, KDim)
    z_exner_ex_pr = zero_field(mesh, CellDim, KDim)

    z_exner_ex_pr_ref, exner_pr_ref = mo_solve_nonhydro_stencil_02_numpy(
        np.asarray(exner),
        np.asarray(exner_ref_mc),
        np.asarray(exner_pr),
        np.asarray(exner_exfac),
    )
    mo_solve_nonhydro_stencil_02(
        exner_exfac, exner, exner_ref_mc, exner_pr, z_exner_ex_pr, offset_provider={}
    )
    assert np.allclose(z_exner_ex_pr_ref, z_exner_ex_pr)
    assert np.allclose(exner_pr_ref, exner_pr)
