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
import pytest
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.model.atmosphere.advection.recon_lsq_cell_c_svd_stencil import (
    recon_lsq_cell_c_svd_stencil,
)
from icon4py.model.common.dimension import C2E2C2E2CDim, CECECDim, CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, zero_field


def recon_lsq_cell_c_svd_stencil_numpy(
    c2e2c2e2c: np.ndarray,
    p_cc: np.ndarray,
    lsq_pseudoinv_1: np.ndarray,
    lsq_pseudoinv_2: np.ndarray,
    lsq_pseudoinv_3: np.ndarray,
    lsq_pseudoinv_4: np.ndarray,
    lsq_pseudoinv_5: np.ndarray,
    lsq_pseudoinv_6: np.ndarray,
    lsq_pseudoinv_7: np.ndarray,
    lsq_pseudoinv_8: np.ndarray,
    lsq_pseudoinv_9: np.ndarray,
    lsq_moments_1: np.ndarray,
    lsq_moments_2: np.ndarray,
    lsq_moments_3: np.ndarray,
    lsq_moments_4: np.ndarray,
    lsq_moments_5: np.ndarray,
    lsq_moments_6: np.ndarray,
    lsq_moments_7: np.ndarray,
    lsq_moments_8: np.ndarray,
    lsq_moments_9: np.ndarray,
) -> tuple[np.ndarray]:

    lsq_moments_1 = np.expand_dims(lsq_moments_1, axis=-1)
    lsq_moments_2 = np.expand_dims(lsq_moments_2, axis=-1)
    lsq_moments_3 = np.expand_dims(lsq_moments_3, axis=-1)
    lsq_moments_4 = np.expand_dims(lsq_moments_4, axis=-1)
    lsq_moments_5 = np.expand_dims(lsq_moments_5, axis=-1)
    lsq_moments_6 = np.expand_dims(lsq_moments_6, axis=-1)
    lsq_moments_7 = np.expand_dims(lsq_moments_7, axis=-1)
    lsq_moments_8 = np.expand_dims(lsq_moments_8, axis=-1)
    lsq_moments_9 = np.expand_dims(lsq_moments_9, axis=-1)
    lsq_moments_1 = np.broadcast_to(lsq_moments_1, p_cc.shape)
    lsq_moments_2 = np.broadcast_to(lsq_moments_2, p_cc.shape)
    lsq_moments_3 = np.broadcast_to(lsq_moments_3, p_cc.shape)
    lsq_moments_4 = np.broadcast_to(lsq_moments_4, p_cc.shape)
    lsq_moments_5 = np.broadcast_to(lsq_moments_5, p_cc.shape)
    lsq_moments_6 = np.broadcast_to(lsq_moments_6, p_cc.shape)
    lsq_moments_7 = np.broadcast_to(lsq_moments_7, p_cc.shape)
    lsq_moments_8 = np.broadcast_to(lsq_moments_8, p_cc.shape)
    lsq_moments_9 = np.broadcast_to(lsq_moments_9, p_cc.shape)
    lsq_pseudoinv_9 = np.expand_dims(lsq_pseudoinv_9, axis=-1)
    lsq_pseudoinv_8 = np.expand_dims(lsq_pseudoinv_8, axis=-1)
    lsq_pseudoinv_7 = np.expand_dims(lsq_pseudoinv_7, axis=-1)
    lsq_pseudoinv_6 = np.expand_dims(lsq_pseudoinv_6, axis=-1)
    lsq_pseudoinv_5 = np.expand_dims(lsq_pseudoinv_5, axis=-1)
    lsq_pseudoinv_4 = np.expand_dims(lsq_pseudoinv_4, axis=-1)
    lsq_pseudoinv_3 = np.expand_dims(lsq_pseudoinv_3, axis=-1)
    lsq_pseudoinv_2 = np.expand_dims(lsq_pseudoinv_2, axis=-1)
    lsq_pseudoinv_1 = np.expand_dims(lsq_pseudoinv_1, axis=-1)

    p_coeff_10_dsl = (
        lsq_pseudoinv_9[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_9[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_9[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_9[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_9[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_9[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_9[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_9[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_9[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_9_dsl = (
        lsq_pseudoinv_8[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_8[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_8[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_8[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_8[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_8[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_8[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_8[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_8[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_8_dsl = (
        lsq_pseudoinv_7[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_7[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_7[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_7[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_7[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_7[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_7[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_7[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_7[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_7_dsl = (
        lsq_pseudoinv_6[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_6[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_6[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_6[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_6[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_6[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_6[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_6[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_6[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_6_dsl = (
        lsq_pseudoinv_5[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_5[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_5[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_5[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_5[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_5[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_5[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_5[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_5[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_5_dsl = (
        lsq_pseudoinv_4[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_4[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_4[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_4[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_4[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_4[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_4[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_4[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_4[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_4_dsl = (
        lsq_pseudoinv_3[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_3[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_3[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_3[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_3[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_3[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_3[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_3[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_3[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_3_dsl = (
        lsq_pseudoinv_2[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_2[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_2[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_2[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_2[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_2[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_2[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_2[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_2[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_2_dsl = (
        lsq_pseudoinv_1[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
        + lsq_pseudoinv_1[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
        + lsq_pseudoinv_1[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
        + lsq_pseudoinv_1[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
        + lsq_pseudoinv_1[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
        + lsq_pseudoinv_1[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
        + lsq_pseudoinv_1[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
        + lsq_pseudoinv_1[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
        + lsq_pseudoinv_1[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
    )

    p_coeff_1_dsl = p_cc - (
        p_coeff_2_dsl * lsq_moments_1
        + p_coeff_3_dsl * lsq_moments_2
        + p_coeff_4_dsl * lsq_moments_3
        + p_coeff_5_dsl * lsq_moments_4
        + p_coeff_6_dsl * lsq_moments_5
        + p_coeff_7_dsl * lsq_moments_6
        + p_coeff_8_dsl * lsq_moments_7
        + p_coeff_9_dsl * lsq_moments_8
        + p_coeff_10_dsl * lsq_moments_9
    )
    return (
        p_coeff_1_dsl,
        p_coeff_2_dsl,
        p_coeff_3_dsl,
        p_coeff_4_dsl,
        p_coeff_5_dsl,
        p_coeff_6_dsl,
        p_coeff_7_dsl,
        p_coeff_8_dsl,
        p_coeff_9_dsl,
        p_coeff_10_dsl,
    )


@pytest.mark.slow_tests
def test_recon_lsq_cell_c_svd_stencil():
    grid = SimpleGrid()
    p_cc = random_field(grid, CellDim, KDim)
    lsq_pseudoinv_1 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_2 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_3 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_4 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_5 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_6 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_7 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_8 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_9 = random_field(grid, CellDim, C2E2C2E2CDim)
    lsq_pseudoinv_1_field = as_1D_sparse_field(lsq_pseudoinv_1, CECECDim)
    lsq_pseudoinv_2_field = as_1D_sparse_field(lsq_pseudoinv_2, CECECDim)
    lsq_pseudoinv_3_field = as_1D_sparse_field(lsq_pseudoinv_3, CECECDim)
    lsq_pseudoinv_4_field = as_1D_sparse_field(lsq_pseudoinv_4, CECECDim)
    lsq_pseudoinv_5_field = as_1D_sparse_field(lsq_pseudoinv_5, CECECDim)
    lsq_pseudoinv_6_field = as_1D_sparse_field(lsq_pseudoinv_6, CECECDim)
    lsq_pseudoinv_7_field = as_1D_sparse_field(lsq_pseudoinv_7, CECECDim)
    lsq_pseudoinv_8_field = as_1D_sparse_field(lsq_pseudoinv_8, CECECDim)
    lsq_pseudoinv_9_field = as_1D_sparse_field(lsq_pseudoinv_9, CECECDim)
    lsq_moments_1 = random_field(grid, CellDim)
    lsq_moments_2 = random_field(grid, CellDim)
    lsq_moments_3 = random_field(grid, CellDim)
    lsq_moments_4 = random_field(grid, CellDim)
    lsq_moments_5 = random_field(grid, CellDim)
    lsq_moments_6 = random_field(grid, CellDim)
    lsq_moments_7 = random_field(grid, CellDim)
    lsq_moments_8 = random_field(grid, CellDim)
    lsq_moments_9 = random_field(grid, CellDim)
    p_coeff_1_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_2_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_3_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_4_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_5_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_6_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_7_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_8_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_9_dsl = zero_field(grid, CellDim, KDim)
    p_coeff_10_dsl = zero_field(grid, CellDim, KDim)

    (
        ref_1,
        ref_2,
        ref_3,
        ref_4,
        ref_5,
        ref_6,
        ref_7,
        ref_8,
        ref_9,
        ref_10,
    ) = recon_lsq_cell_c_svd_stencil_numpy(
        grid.connectivities[C2E2C2E2CDim],
        np.asarray(p_cc),
        np.asarray(lsq_pseudoinv_1),
        np.asarray(lsq_pseudoinv_2),
        np.asarray(lsq_pseudoinv_3),
        np.asarray(lsq_pseudoinv_4),
        np.asarray(lsq_pseudoinv_5),
        np.asarray(lsq_pseudoinv_6),
        np.asarray(lsq_pseudoinv_7),
        np.asarray(lsq_pseudoinv_8),
        np.asarray(lsq_pseudoinv_9),
        np.asarray(lsq_moments_1),
        np.asarray(lsq_moments_2),
        np.asarray(lsq_moments_3),
        np.asarray(lsq_moments_4),
        np.asarray(lsq_moments_5),
        np.asarray(lsq_moments_6),
        np.asarray(lsq_moments_7),
        np.asarray(lsq_moments_8),
        np.asarray(lsq_moments_9),
    )

    recon_lsq_cell_c_svd_stencil(
        p_cc,
        lsq_pseudoinv_1_field,
        lsq_pseudoinv_2_field,
        lsq_pseudoinv_3_field,
        lsq_pseudoinv_4_field,
        lsq_pseudoinv_5_field,
        lsq_pseudoinv_6_field,
        lsq_pseudoinv_7_field,
        lsq_pseudoinv_8_field,
        lsq_pseudoinv_9_field,
        lsq_moments_1,
        lsq_moments_2,
        lsq_moments_3,
        lsq_moments_4,
        lsq_moments_5,
        lsq_moments_6,
        lsq_moments_7,
        lsq_moments_8,
        lsq_moments_9,
        p_coeff_1_dsl,
        p_coeff_2_dsl,
        p_coeff_3_dsl,
        p_coeff_4_dsl,
        p_coeff_5_dsl,
        p_coeff_6_dsl,
        p_coeff_7_dsl,
        p_coeff_8_dsl,
        p_coeff_9_dsl,
        p_coeff_10_dsl,
        offset_provider={
            "C2E2C2E2C": grid.get_c2e2c2e2c_offset_provider(),
            "C2CECEC": StridedNeighborOffsetProvider(CellDim, CECECDim, grid.size[C2E2C2E2CDim]),
        },
    )
    co1 = np.asarray(p_coeff_1_dsl)
    co2 = np.asarray(p_coeff_2_dsl)
    co3 = np.asarray(p_coeff_3_dsl)
    co4 = np.asarray(p_coeff_4_dsl)
    co5 = np.asarray(p_coeff_5_dsl)
    co6 = np.asarray(p_coeff_6_dsl)
    co7 = np.asarray(p_coeff_7_dsl)
    co8 = np.asarray(p_coeff_8_dsl)
    co9 = np.asarray(p_coeff_9_dsl)
    co10 = np.asarray(p_coeff_10_dsl)
    assert np.allclose(ref_1, co1)
    assert np.allclose(ref_2, co2)
    assert np.allclose(ref_3, co3)
    assert np.allclose(ref_4, co4)
    assert np.allclose(ref_5, co5)
    assert np.allclose(ref_6, co6)
    assert np.allclose(ref_7, co7)
    assert np.allclose(ref_8, co8)
    assert np.allclose(ref_9, co9)
    assert np.allclose(ref_10, co10)
