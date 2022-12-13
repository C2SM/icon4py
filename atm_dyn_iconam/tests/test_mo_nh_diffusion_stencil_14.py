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
from functional.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_14 import (
    mo_nh_diffusion_stencil_14,
)
from icon4py.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_nh_diffusion_stencil_14_numpy(
    c2e: np.array, z_nabla2_e: np.array, geofac_div: np.array
) -> np.array:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=1)  # sum along edge dimension
    return z_temp


def test_mo_nh_diffusion_stencil_14():
    mesh = SimpleMesh()

    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    geofac_div = random_field(mesh, CellDim, C2EDim)

    out = zero_field(mesh, CellDim, KDim)

    ref = mo_nh_diffusion_stencil_14_numpy(
        mesh.c2e, np.asarray(z_nabla2_e), np.asarray(geofac_div)
    )
    mo_nh_diffusion_stencil_14(
        z_nabla2_e,
        geofac_div,
        out,
        offset_provider={
            "C2E": mesh.get_c2e_offset_provider(),
            "C2CE": StridedNeighborOffsetProvider(CellDim, CEDim, mesh.n_c2e),
            "C2EDim": C2EDim,
            "CEDim": CEDim,
        },
    )
    assert np.allclose(out, ref)
