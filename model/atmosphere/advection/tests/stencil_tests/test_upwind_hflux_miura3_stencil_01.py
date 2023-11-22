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

from icon4py.model.atmosphere.advection.upwind_hflux_miura3_stencil_01 import (
    upwind_hflux_miura3_stencil_01,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, random_mask, zero_field


def upwind_hflux_miura3_stencil_01_numpy(
    e2c: np.array,
    z_lsq_coeff_1: np.array,
    z_lsq_coeff_2: np.array,
    z_lsq_coeff_3: np.array,
    z_lsq_coeff_4: np.array,
    z_lsq_coeff_5: np.array,
    z_lsq_coeff_6: np.array,
    z_lsq_coeff_7: np.array,
    z_lsq_coeff_8: np.array,
    z_lsq_coeff_9: np.array,
    z_lsq_coeff_10: np.array,
    z_quad_vector_sum_1: np.array,
    z_quad_vector_sum_2: np.array,
    z_quad_vector_sum_3: np.array,
    z_quad_vector_sum_4: np.array,
    z_quad_vector_sum_5: np.array,
    z_quad_vector_sum_6: np.array,
    z_quad_vector_sum_7: np.array,
    z_quad_vector_sum_8: np.array,
    z_quad_vector_sum_9: np.array,
    z_quad_vector_sum_10: np.array,
    z_dreg_area: np.array,
    p_mass_flx_e: np.array,
    cell_rel_idx_dsl: np.array,
) -> np.array:

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

    p_out_e_miura3 = (
        (
            np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_1_e2c[:, 1],
                z_lsq_coeff_1_e2c[:, 0],
            )
            * z_quad_vector_sum_1
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_2_e2c[:, 1],
                z_lsq_coeff_2_e2c[:, 0],
            )
            * z_quad_vector_sum_2
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_3_e2c[:, 1],
                z_lsq_coeff_3_e2c[:, 0],
            )
            * z_quad_vector_sum_3
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_4_e2c[:, 1],
                z_lsq_coeff_4_e2c[:, 0],
            )
            * z_quad_vector_sum_4
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_5_e2c[:, 1],
                z_lsq_coeff_5_e2c[:, 0],
            )
            * z_quad_vector_sum_5
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_6_e2c[:, 1],
                z_lsq_coeff_6_e2c[:, 0],
            )
            * z_quad_vector_sum_6
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_7_e2c[:, 1],
                z_lsq_coeff_7_e2c[:, 0],
            )
            * z_quad_vector_sum_7
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_8_e2c[:, 1],
                z_lsq_coeff_8_e2c[:, 0],
            )
            * z_quad_vector_sum_8
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_9_e2c[:, 1],
                z_lsq_coeff_9_e2c[:, 0],
            )
            * z_quad_vector_sum_9
            + np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_10_e2c[:, 1],
                z_lsq_coeff_10_e2c[:, 0],
            )
            * z_quad_vector_sum_10
        )
        / z_dreg_area
        * p_mass_flx_e
    )

    return p_out_e_miura3


def test_upwind_hflux_miura3_stencil_01(backend):
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
    z_quad_vector_sum_1 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_2 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_3 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_4 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_5 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_6 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_7 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_8 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_9 = random_field(grid, EdgeDim, KDim)
    z_quad_vector_sum_10 = random_field(grid, EdgeDim, KDim)
    p_mass_flx_e = random_field(grid, EdgeDim, KDim)
    z_dreg_area = random_field(grid, EdgeDim, KDim)
    cell_rel_idx_dsl = random_mask(grid, EdgeDim, KDim, dtype=int32)
    p_out_e_miura3 = zero_field(grid, EdgeDim, KDim)

    ref = upwind_hflux_miura3_stencil_01_numpy(
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
        np.asarray(z_quad_vector_sum_1),
        np.asarray(z_quad_vector_sum_2),
        np.asarray(z_quad_vector_sum_3),
        np.asarray(z_quad_vector_sum_4),
        np.asarray(z_quad_vector_sum_5),
        np.asarray(z_quad_vector_sum_6),
        np.asarray(z_quad_vector_sum_7),
        np.asarray(z_quad_vector_sum_8),
        np.asarray(z_quad_vector_sum_9),
        np.asarray(z_quad_vector_sum_10),
        np.asarray(z_dreg_area),
        np.asarray(p_mass_flx_e),
        np.asarray(cell_rel_idx_dsl),
    )

    upwind_hflux_miura3_stencil_01.with_backend(backend)(
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
        z_quad_vector_sum_1,
        z_quad_vector_sum_2,
        z_quad_vector_sum_3,
        z_quad_vector_sum_4,
        z_quad_vector_sum_5,
        z_quad_vector_sum_6,
        z_quad_vector_sum_7,
        z_quad_vector_sum_8,
        z_quad_vector_sum_9,
        z_quad_vector_sum_10,
        z_dreg_area,
        p_mass_flx_e,
        cell_rel_idx_dsl,
        p_out_e_miura3,
        offset_provider={
            "E2C": grid.get_offset_provider("E2C"),
        },
    )
    assert np.allclose(np.asarray(p_out_e_miura3), ref)
