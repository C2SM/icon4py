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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_18 import (
    _mo_solve_nonhydro_stencil_18,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_18_numpy(
    e2c: np.array,
    inv_dual_edge_length: np.array,
    z_exner_ex_pr: np.array,
) -> np.array:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    z_exner_ex_pr_e2c = z_exner_ex_pr[e2c]
    z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

    z_gradh_exner = inv_dual_edge_length * z_exner_ex_weighted
    return z_gradh_exner


def test_mo_solve_nonhydro_stencil_18():
    mesh = SimpleMesh()

    inv_dual_edge_length = random_field(mesh, EdgeDim)
    z_exner_ex_pr = random_field(mesh, CellDim, KDim)
    z_gradh_exner = random_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_18_numpy(
        mesh.e2c,
        np.asarray(inv_dual_edge_length),
        np.asarray(z_exner_ex_pr),
    )

    _mo_solve_nonhydro_stencil_18(
        inv_dual_edge_length,
        z_exner_ex_pr,
        z_gradh_exner,
        out=z_gradh_exner,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(z_gradh_exner, ref)
