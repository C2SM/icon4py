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

from icon4py.advection.step_advection_stencil_04 import step_advection_stencil_04
from icon4py.common.dimension import CellDim, KDim
from .test_utils.simple_mesh import SimpleMesh
from .test_utils.helpers import random_field, zero_field


def step_advection_stencil_04_numpy(
    p_tracer_now: np.array,
    p_tracer_new: np.array,
    p_dtime,
) -> np.array:
    opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime
    return opt_ddt_tracer_adv


def test_step_advection_stencil_04():
    mesh = SimpleMesh()

    p_tracer_now = random_field(mesh, CellDim, KDim)
    p_tracer_new = random_field(mesh, CellDim, KDim)
    opt_ddt_tracer_adv = zero_field(mesh, CellDim, KDim)
    p_dtime = np.float64(5.0)

    ref = step_advection_stencil_04_numpy(
        np.asarray(p_tracer_now),
        np.asarray(p_tracer_new),
        p_dtime,
    )
    step_advection_stencil_04(
        p_tracer_now,
        p_tracer_new,
        opt_ddt_tracer_adv,
        p_dtime,
        offset_provider={},
    )
    assert np.allclose(opt_ddt_tracer_adv, ref)
