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


from typing import Tuple

import numpy as np

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_67 import (
    mo_solve_nonhydro_stencil_67,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_67_theta_v_numpy(exner: np.array) -> np.array:

    theta_v = np.copy(exner)
    return theta_v


def mo_solve_nonhydro_stencil_67_exner_numpy(
    rho: np.array, theta_v: np.array, rd_o_cvd: float, rd_o_p0ref: float
) -> np.array:

    exner = np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * theta_v))
    return exner


def mo_solve_nonhydro_stencil_67_numpy(
    rho: np.array, exner: np.array, rd_o_cvd: float, rd_o_p0ref: float
) -> Tuple[np.array]:

    theta_v = mo_solve_nonhydro_stencil_67_theta_v_numpy(exner)
    exner = mo_solve_nonhydro_stencil_67_exner_numpy(rho, theta_v, rd_o_cvd, rd_o_p0ref)

    return theta_v, exner


def test_mo_solve_nonhydro_stencil_67():
    mesh = SimpleMesh()

    rd_o_cvd = np.float64(10.0)
    rd_o_p0ref = np.float64(20.0)
    rho = random_field(mesh, CellDim, KDim, low=1, high=2)
    theta_v = random_field(mesh, CellDim, KDim, low=1, high=2)
    exner = random_field(mesh, CellDim, KDim, low=1, high=2)

    theta_v_ref, exner_ref = mo_solve_nonhydro_stencil_67_numpy(
        np.asarray(rho), np.asarray(exner), rd_o_cvd, rd_o_p0ref
    )

    mo_solve_nonhydro_stencil_67(
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        offset_provider={},
    )

    assert np.allclose(theta_v, theta_v_ref)
    assert np.allclose(exner, exner_ref)
