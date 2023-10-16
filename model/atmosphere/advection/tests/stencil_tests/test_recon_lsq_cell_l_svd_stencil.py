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
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.model.atmosphere.advection.recon_lsq_cell_l_svd_stencil import (
    recon_lsq_cell_l_svd_stencil,
)
from icon4py.model.common.dimension import C2E2CDim, CECDim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def recon_lsq_cell_l_svd_stencil_numpy(
    c2e2c: np.ndarray,
    p_cc: np.ndarray,
    lsq_pseudoinv_1: np.ndarray,
    lsq_pseudoinv_2: np.ndarray,
) -> tuple[np.ndarray]:
    p_cc_e = np.expand_dims(p_cc, axis=1)
    n_diff = p_cc[c2e2c] - p_cc_e
    lsq_pseudoinv_2 = np.expand_dims(lsq_pseudoinv_2, axis=-1)
    lsq_pseudoinv_1 = np.expand_dims(lsq_pseudoinv_1, axis=-1)
    p_coeff_1 = p_cc
    p_coeff_2 = np.sum(lsq_pseudoinv_1 * n_diff, axis=1)
    p_coeff_3 = np.sum(lsq_pseudoinv_2 * n_diff, axis=1)
    return p_coeff_1, p_coeff_2, p_coeff_3


def test_recon_lsq_cell_l_svd_stencil():
    mesh = SimpleMesh()
    p_cc = random_field(mesh, CellDim, KDim)
    lsq_pseudoinv_1 = random_field(mesh, CellDim, C2E2CDim)
    lsq_pseudoinv_1_field = as_1D_sparse_field(lsq_pseudoinv_1, CECDim)

    lsq_pseudoinv_2 = random_field(mesh, CellDim, C2E2CDim)
    lsq_pseudoinv_2_field = as_1D_sparse_field(lsq_pseudoinv_2, CECDim)
    p_coeff_1 = zero_field(mesh, CellDim, KDim)
    p_coeff_2 = zero_field(mesh, CellDim, KDim)
    p_coeff_3 = zero_field(mesh, CellDim, KDim)

    ref_1, ref_2, ref_3 = recon_lsq_cell_l_svd_stencil_numpy(
        mesh.c2e2c,
        np.asarray(p_cc),
        np.asarray(lsq_pseudoinv_1),
        np.asarray(lsq_pseudoinv_2),
    )

    recon_lsq_cell_l_svd_stencil(
        p_cc,
        lsq_pseudoinv_1_field,
        lsq_pseudoinv_2_field,
        p_coeff_1,
        p_coeff_2,
        p_coeff_3,
        offset_provider={
            "C2E2C": mesh.get_c2e2c_offset_provider(),
            "C2CEC": StridedNeighborOffsetProvider(CellDim, CECDim, mesh.n_c2e2c),
        },
    )
    co1 = np.asarray(p_coeff_1)
    co2 = np.asarray(p_coeff_2)
    co3 = np.asarray(p_coeff_3)
    assert np.allclose(ref_1, co1)
    assert np.allclose(ref_2, co2)
    assert np.allclose(ref_3, co3)
