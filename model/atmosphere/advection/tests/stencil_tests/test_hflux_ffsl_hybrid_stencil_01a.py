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

from icon4py.model.atmosphere.advection.hflux_ffsl_hybrid_stencil_01a import (
    hflux_ffsl_hybrid_stencil_01a,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import constant_field, random_field, zero_field


def hflux_ffsl_hybrid_stencil_01a_numpy(
    e2c: np.array,
    z_lsq_coeff_1: np.ndarray,
    z_lsq_coeff_2: np.ndarray,
    z_lsq_coeff_3: np.ndarray,
    z_lsq_coeff_4: np.ndarray,
    z_lsq_coeff_5: np.ndarray,
    z_lsq_coeff_6: np.ndarray,
    z_lsq_coeff_7: np.ndarray,
    z_lsq_coeff_8: np.ndarray,
    z_lsq_coeff_9: np.ndarray,
    z_lsq_coeff_10: np.ndarray,
    z_quad_vector_sum0_1: np.ndarray,
    z_quad_vector_sum0_2: np.ndarray,
    z_quad_vector_sum0_3: np.ndarray,
    z_quad_vector_sum0_4: np.ndarray,
    z_quad_vector_sum0_5: np.ndarray,
    z_quad_vector_sum0_6: np.ndarray,
    z_quad_vector_sum0_7: np.ndarray,
    z_quad_vector_sum0_8: np.ndarray,
    z_quad_vector_sum0_9: np.ndarray,
    z_quad_vector_sum0_10: np.ndarray,
    patch0_cell_rel_idx_dsl: np.ndarray,
):

    z_lsq_coeff_1_e2c = z_lsq_coeff_1[e2c]
    z_lsq_coeff_2_e2c = z_lsq_coeff_2[e2c]
    z_lsq_coeff_3_e2c = z_lsq_coeff_3[e2c]
    z_lsq_coeff_4_e2c = z_lsq_coeff_4[e2c]
    z_lsq_coeff_5_e2c = z_lsq_coeff_5[e2c]
    z_lsq_coeff_6_e2c = z_lsq_coeff_6[e2c]
    z_lsq_coeff_7_e2c = z_lsq_coeff_7[e2c]
    z_lsq_coeff_8_e2c = z_lsq_coeff_8[e2c]
    z_lsq_coeff_9_e2c = z_lsq_coeff_9[e2c]
    z_lsq_coeff_10_e2c = z_lsq_coeff_10[e2c]

    p_out_e_hybrid_1a = (
        np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_1_e2c[:, 1],
            z_lsq_coeff_1_e2c[:, 0],
        )
        * z_quad_vector_sum0_1
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_2_e2c[:, 1],
            z_lsq_coeff_2_e2c[:, 0],
        )
        * z_quad_vector_sum0_2
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_3_e2c[:, 1],
            z_lsq_coeff_3_e2c[:, 0],
        )
        * z_quad_vector_sum0_3
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_4_e2c[:, 1],
            z_lsq_coeff_4_e2c[:, 0],
        )
        * z_quad_vector_sum0_4
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_5_e2c[:, 1],
            z_lsq_coeff_5_e2c[:, 0],
        )
        * z_quad_vector_sum0_5
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_6_e2c[:, 1],
            z_lsq_coeff_6_e2c[:, 0],
        )
        * z_quad_vector_sum0_6
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_7_e2c[:, 1],
            z_lsq_coeff_7_e2c[:, 0],
        )
        * z_quad_vector_sum0_7
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_8_e2c[:, 1],
            z_lsq_coeff_8_e2c[:, 0],
        )
        * z_quad_vector_sum0_8
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_9_e2c[:, 1],
            z_lsq_coeff_9_e2c[:, 0],
        )
        * z_quad_vector_sum0_9
        + np.where(
            patch0_cell_rel_idx_dsl == int32(1),
            z_lsq_coeff_10_e2c[:, 1],
            z_lsq_coeff_10_e2c[:, 0],
        )
        * z_quad_vector_sum0_10
    )

    return p_out_e_hybrid_1a


def test_hflux_ffsl_hybrid_stencil_01a():
    grid = SimpleGrid()
    z_lsq_coeff_1 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_2 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_3 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_4 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_5 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_6 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_7 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_8 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_9 = random_field(grid, CellDim, KDim)
    z_lsq_coeff_10 = random_field(grid, CellDim, KDim)
    z_quad_vector_sum0_1 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_2 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_3 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_4 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_5 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_6 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_7 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_8 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_9 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum0_10 = random_field(grid, EdgeDim, KDim)
    patch0_cell_rel_idx_dsl = constant_field(grid, 1, EdgeDim, KDim, dtype=int32)
    p_out_e_hybrid_1a = zero_field(grid, EdgeDim, KDim)

    ref = hflux_ffsl_hybrid_stencil_01a_numpy(
        grid.connectivities[E2CDim],
        np.asarray(z_lsq_coeff_1),
        np.asarray(z_lsq_coeff_2),
        np.asarray(z_lsq_coeff_3),
        np.asarray(z_lsq_coeff_4),
        np.asarray(z_lsq_coeff_5),
        np.asarray(z_lsq_coeff_6),
        np.asarray(z_lsq_coeff_7),
        np.asarray(z_lsq_coeff_8),
        np.asarray(z_lsq_coeff_9),
        np.asarray(z_lsq_coeff_10),
        np.asarray(z_quad_vector_sum0_1),
        np.asarray(z_quad_vector_sum0_2),
        np.asarray(z_quad_vector_sum0_3),
        np.asarray(z_quad_vector_sum0_4),
        np.asarray(z_quad_vector_sum0_5),
        np.asarray(z_quad_vector_sum0_6),
        np.asarray(z_quad_vector_sum0_7),
        np.asarray(z_quad_vector_sum0_8),
        np.asarray(z_quad_vector_sum0_9),
        np.asarray(z_quad_vector_sum0_10),
        np.asarray(patch0_cell_rel_idx_dsl),
    )

    hflux_ffsl_hybrid_stencil_01a(
        z_lsq_coeff_1,
        z_lsq_coeff_2,
        z_lsq_coeff_3,
        z_lsq_coeff_4,
        z_lsq_coeff_5,
        z_lsq_coeff_6,
        z_lsq_coeff_7,
        z_lsq_coeff_8,
        z_lsq_coeff_9,
        z_lsq_coeff_10,
        z_quad_vector_sum0_1,
        z_quad_vector_sum0_2,
        z_quad_vector_sum0_3,
        z_quad_vector_sum0_4,
        z_quad_vector_sum0_5,
        z_quad_vector_sum0_6,
        z_quad_vector_sum0_7,
        z_quad_vector_sum0_8,
        z_quad_vector_sum0_9,
        z_quad_vector_sum0_10,
        patch0_cell_rel_idx_dsl,
        p_out_e_hybrid_1a,
        offset_provider={
            "E2C": grid.get_offset_provider["E2C"],
        },
    )

    assert np.allclose(p_out_e_hybrid_1a, ref)
