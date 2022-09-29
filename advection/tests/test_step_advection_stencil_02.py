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

from icon4py.advection.step_advection_stencil_02 import step_advection_stencil_02
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def step_advection_stencil_02_numpy(
    rhodz_new: np.ndarray,
    p_mflx_contra_v: np.ndarray,
    deepatmo_divzl: np.ndarray,
    deepatmo_divzu: np.ndarray,
    pd_time: float,
) -> np.ndarray:

    tmp = (
        np.roll(p_mflx_contra_v, axis=1, shift=-1) * deepatmo_divzl
        - p_mflx_contra_v * deepatmo_divzu
    )
    return np.maximum(0.1 * rhodz_new, rhodz_new) -pd_time * tmp


def test_step_advection_stencil_02():
    mesh = SimpleMesh()
    rhodz_ast = random_field(mesh, CellDim, KDim)
    p_mflx_contra = random_field(mesh, CellDim, KDim)
    deepatmo_divzl = random_field(mesh, KDim)
    deepatmo_divzu = random_field(mesh, KDim)
    result = zero_field(mesh, CellDim, KDim)
    p_dtime = 0.1

    ref = step_advection_stencil_02_numpy(
        np.asarray(rhodz_ast),
        np.asarray(p_mflx_contra),
        np.asarray(deepatmo_divzl),
        np.asarray(deepatmo_divzu),
        p_dtime,
    )

    step_advection_stencil_02(
        rhodz_ast,
        p_mflx_contra,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        result,
        offset_provider={"Koff": KDim},
    )

    result1 = np.asarray(result)[:, :-1]
    assert np.allclose(ref[:, :-1], result1)
