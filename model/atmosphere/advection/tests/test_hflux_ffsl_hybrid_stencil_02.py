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
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator import embedded as it_embedded

from icon4py.model.atmosphere.advection.hflux_ffsl_hybrid_stencil_02 import hflux_ffsl_hybrid_stencil_02
from icon4py.model.common.dimension import KDim, EdgeDim

from icon4py.model.common.test_utils.helpers import _shape, random_field, zero_field, constant_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def hflux_ffsl_hybrid_stencil_02_numpy(
    p_out_e_hybrid_2: np.ndarray,
    p_mass_flx_e: np.ndarray,
    z_dreg_area: np.ndarray,
):

    p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

    return p_out_e_hybrid_2


def test_hflux_ffsl_hybrid_stencil_02():
    mesh = SimpleMesh()
    p_out_e_hybrid_2 = random_field(mesh, EdgeDim, KDim)
    p_mass_flx_e = random_field(mesh, EdgeDim, KDim)
    z_dreg_area = random_field(mesh, EdgeDim, KDim)

    ref = hflux_ffsl_hybrid_stencil_02_numpy(
        np.asarray(p_out_e_hybrid_2),
        np.asarray(p_mass_flx_e),
        np.asarray(z_dreg_area),
    )

    hflux_ffsl_hybrid_stencil_02(
        p_out_e_hybrid_2,
        p_mass_flx_e,
        z_dreg_area,
        offset_provider={},
    )

    assert np.allclose(p_out_e_hybrid_2, ref)
