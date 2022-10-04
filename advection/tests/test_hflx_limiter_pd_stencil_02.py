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

from icon4py.advection.hflx_limiter_pd_stencil_02 import (
    hflx_limiter_pd_stencil_02,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def hflx_limiter_pd_stencil_02_numpy(
    e2c: np.array,
    refin_ctrl: np.array,
    r_m: np.array,
    p_mflx_tracer_h_in: np.array,
    bound,
):
    r_m_e2c = r_m[e2c]
    refin_ctrl = np.expand_dims(refin_ctrl, axis=-1)
    p_mflx_tracer_h_out = np.where(
        refin_ctrl != bound,
        np.where(
            p_mflx_tracer_h_in >= 0,
            p_mflx_tracer_h_in * r_m_e2c[:, 0],
            p_mflx_tracer_h_in * r_m_e2c[:, 1],
        ),
        p_mflx_tracer_h_in,
    )
    return p_mflx_tracer_h_out


def test_hflx_limiter_pd_stencil_02():
    mesh = SimpleMesh()

    refin_ctrl = random_field(mesh, EdgeDim)
    r_m = random_field(mesh, CellDim, KDim)
    p_mflx_tracer_h_in = random_field(mesh, EdgeDim, KDim)
    p_mflx_tracer_h_out = random_field(mesh, EdgeDim, KDim)
    bound = np.float64(7.0)

    ref = hflx_limiter_pd_stencil_02_numpy(
        mesh.e2c,
        np.asarray(refin_ctrl),
        np.asarray(r_m),
        np.asarray(p_mflx_tracer_h_in),
        bound,
    )

    hflx_limiter_pd_stencil_02(
        refin_ctrl,
        r_m,
        p_mflx_tracer_h_in,
        p_mflx_tracer_h_out,
        bound,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(p_mflx_tracer_h_out, ref)
