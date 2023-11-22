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

from icon4py.model.atmosphere.advection.hflux_ffsl_hybrid_stencil_02 import (
    hflux_ffsl_hybrid_stencil_02,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field


def hflux_ffsl_hybrid_stencil_02_numpy(
    p_out_e_hybrid_2: np.ndarray,
    p_mass_flx_e: np.ndarray,
    z_dreg_area: np.ndarray,
):

    p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

    return p_out_e_hybrid_2


def test_hflux_ffsl_hybrid_stencil_02(backend):
    grid = SimpleGrid()
    p_out_e_hybrid_2 = random_field(grid, EdgeDim, KDim)
    p_mass_flx_e = random_field(grid, EdgeDim, KDim)
    z_dreg_area = random_field(grid, EdgeDim, KDim)

    ref = hflux_ffsl_hybrid_stencil_02_numpy(
        np.asarray(p_out_e_hybrid_2),
        np.asarray(p_mass_flx_e),
        np.asarray(z_dreg_area),
    )

    hflux_ffsl_hybrid_stencil_02.with_backend(backend)(
        p_out_e_hybrid_2,
        p_mass_flx_e,
        z_dreg_area,
        offset_provider={},
    )

    assert np.allclose(p_out_e_hybrid_2, ref)
