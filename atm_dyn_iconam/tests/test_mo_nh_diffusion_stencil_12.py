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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_12 import (
    mo_nh_diffusion_stencil_12,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_nh_diffusion_stencil_12_numpy(
    e2c: np.array,
    kh_smag_e: np.array,
    enh_diffu_3d: np.array,
) -> np.array:
    kh_smag_e = np.maximum(kh_smag_e, np.max(enh_diffu_3d[e2c], axis=1))
    return kh_smag_e


def test_mo_nh_diffusion_stencil_12():
    mesh = SimpleMesh()
    kh_smag_e = random_field(mesh, EdgeDim, KDim)
    enh_diffu_3d = random_field(mesh, CellDim, KDim)

    kh_smag_e_ref = mo_nh_diffusion_stencil_12_numpy(
        mesh.e2c, np.asarray(kh_smag_e), np.asarray(enh_diffu_3d)
    )

    mo_nh_diffusion_stencil_12(
        kh_smag_e, enh_diffu_3d, offset_provider={"E2C": mesh.get_e2c_offset_provider()}
    )

    np.allclose(kh_smag_e_ref, kh_smag_e)
