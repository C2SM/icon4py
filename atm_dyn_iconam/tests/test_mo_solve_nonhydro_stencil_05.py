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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_05 import (
    mo_solve_nonhydro_stencil_05,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_05_numpy(
    wgtfac_c: np.array,
    z_exner_ex_pr: np.array,
) -> np.array:
    z_exner_ex_pr_offset_1 = np.roll(z_exner_ex_pr, shift=1, axis=1)
    z_exner_ic = wgtfac_c * z_exner_ex_pr + (1.0 - wgtfac_c) * z_exner_ex_pr_offset_1
    return z_exner_ic


def test_mo_solve_nonhydro_stencil_05():
    mesh = SimpleMesh()
    z_exner_ex_pr = random_field(mesh, CellDim, KDim)
    wgtfac_c = random_field(mesh, CellDim, KDim)
    z_exner_ic = zero_field(mesh, CellDim, KDim)

    z_exner_ic_ref = mo_solve_nonhydro_stencil_05_numpy(
        np.asarray(wgtfac_c),
        np.asarray(z_exner_ex_pr),
    )

    mo_solve_nonhydro_stencil_05(
        wgtfac_c,
        z_exner_ex_pr,
        z_exner_ic,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_exner_ic, z_exner_ic_ref)
