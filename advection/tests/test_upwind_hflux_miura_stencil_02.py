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
from upwind_hflux_miura_stencil_02 import upwind_hflux_miura_stencil_02

from icon4py.common.dimension import C2E2CDim, CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def upwind_hflux_miura_stencil_02_numpy(
    c2e2c: np.ndarray,
    p_cc: np.ndarray,
    lsq_pseudoinv_1: np.ndarray,
    lsq_pseudoinv_2: np.ndarray,
) -> tuple[np.ndarray]:
    lsq_pseudoinv_1 = np.expand_dims(lsq_pseudoinv_1, axis=-1)
    lsq_pseudoinv_2 = np.expand_dims(lsq_pseudoinv_2, axis=-1)
    p_coeff_1 = np.sum(lsq_pseudoinv_1 * p_cc[c2e2c], axis=1) - p_cc
    p_coeff_2 = np.sum(lsq_pseudoinv_2 * p_cc[c2e2c], axis=1) - p_cc
    return p_coeff_1, p_coeff_2


def test_upwind_hflux_miura_stencil_02():
    mesh = SimpleMesh()
    p_cc = random_field(mesh, CellDim, KDim)
    lsq_pseudoinv_1 = random_field(mesh, C2E2CDim)
    lsq_pseudoinv_2 = random_field(mesh, C2E2CDim)
    p_coeff_1 = zero_field(mesh, CellDim, KDim)
    p_coeff_2 = zero_field(mesh, CellDim, KDim)

    ref_1, ref_2 = upwind_hflux_miura_stencil_02_numpy(
        mesh.c2e2c,
        np.asarray(p_cc),
        np.asarray(lsq_pseudoinv_1),
        np.asarray(lsq_pseudoinv_2),
    )

    upwind_hflux_miura_stencil_02(
        p_cc,
        lsq_pseudoinv_1,
        lsq_pseudoinv_1,
        p_coeff_1,
        p_coeff_2,
        offset_provider={"C2E2C": mesh.get_c2e2c_offset_provider()},
    )
    np.allclose(ref_1, p_coeff_1)
    np.allclose(ref_2, p_coeff_2)
