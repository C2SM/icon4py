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

from icon4py.model.atmosphere.advection.step_advection_stencil_01 import step_advection_stencil_01
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def step_advection_stencil_01_numpy(
    rhodz_ast: np.ndarray,
    p_mflx_contra_v: np.ndarray,
    deepatmo_divzl: np.ndarray,
    deepatmo_divzu: np.ndarray,
    pd_time: float,
) -> np.ndarray:
    tmp = pd_time * (
        p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
    )
    return rhodz_ast + tmp


def test_step_advection_stencil_01(backend):
    grid = SimpleGrid()
    rhodz_ast = random_field(grid, CellDim, KDim)
    p_mflx_contra = random_field(grid, CellDim, KDim, extend={KDim: 1})
    deepatmo_divzl = random_field(grid, KDim)
    deepatmo_divzu = random_field(grid, KDim)
    result = zero_field(grid, CellDim, KDim)
    p_dtime = 0.1

    ref = step_advection_stencil_01_numpy(
        np.asarray(rhodz_ast),
        np.asarray(p_mflx_contra),
        np.asarray(deepatmo_divzl),
        np.asarray(deepatmo_divzu),
        p_dtime,
    )

    step_advection_stencil_01.with_backend(backend)(
        rhodz_ast,
        p_mflx_contra,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        result,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(ref[:, :-1], result[:, :-1])
