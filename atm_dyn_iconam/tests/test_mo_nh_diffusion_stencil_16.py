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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_16 import (
    mo_nh_diffusion_stencil_16,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_nh_diffusion_stencil_16_numpy(
    z_temp: np.array,
    area: np.array,
    theta_v: np.array,
    exner: np.array,
    rd_o_cvd,
) -> tuple[np.array]:
    area = np.expand_dims(area, axis=0)
    z_theta = theta_v
    theta_v = theta_v + (np.expand_dims(area, axis=-1) * z_temp)
    exner = exner * (1.0 + rd_o_cvd * (theta_v / z_theta - 1.0))
    return theta_v, exner


def test_mo_nh_diffusion_stencil_16():
    mesh = SimpleMesh()

    z_temp = random_field(mesh, CellDim, KDim)
    area = random_field(mesh, CellDim)
    theta_v = random_field(mesh, CellDim, KDim)
    exner = random_field(mesh, CellDim, KDim)
    rd_o_cvd = 5.0

    theta_v_ref, exner_ref = mo_nh_diffusion_stencil_16_numpy(
        np.asarray(z_temp),
        np.asarray(area),
        np.asarray(theta_v),
        np.asarray(exner),
        rd_o_cvd,
    )
    mo_nh_diffusion_stencil_16(
        z_temp,
        area,
        theta_v,
        exner,
        rd_o_cvd,
        offset_provider={},
    )
    assert np.allclose(theta_v, theta_v_ref)
    assert np.allclose(exner, exner_ref)
