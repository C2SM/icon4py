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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _hflux_ffsl_hybrid_stencil_01a(
    z_lsq_coeff_1: Field[[CellDim, KDim], float],
    z_lsq_coeff_2: Field[[CellDim, KDim], float],
    z_lsq_coeff_3: Field[[CellDim, KDim], float],
    z_lsq_coeff_4: Field[[CellDim, KDim], float],
    z_lsq_coeff_5: Field[[CellDim, KDim], float],
    z_lsq_coeff_6: Field[[CellDim, KDim], float],
    z_lsq_coeff_7: Field[[CellDim, KDim], float],
    z_lsq_coeff_8: Field[[CellDim, KDim], float],
    z_lsq_coeff_9: Field[[CellDim, KDim], float],
    z_lsq_coeff_10: Field[[CellDim, KDim], float],
    z_quad_vector_sum0_1: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_2: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_3: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_4: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_5: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_6: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_7: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_8: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_9: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_10: Field[[EdgeDim, KDim], float],
    patch0_cell_rel_idx_dsl: Field[[EdgeDim, KDim], int32],
) -> Field[[EdgeDim, KDim], float]:
    p_out_e_hybrid_1a = (
        where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_1(E2C[1]),
            z_lsq_coeff_1(E2C[0]),
        )
        * z_quad_vector_sum0_1
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_2(E2C[1]),
            z_lsq_coeff_2(E2C[0]),
        )
        * z_quad_vector_sum0_2
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_3(E2C[1]),
            z_lsq_coeff_3(E2C[0]),
        )
        * z_quad_vector_sum0_3
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_4(E2C[1]),
            z_lsq_coeff_4(E2C[0]),
        )
        * z_quad_vector_sum0_4
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_5(E2C[1]),
            z_lsq_coeff_5(E2C[0]),
        )
        * z_quad_vector_sum0_5
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_6(E2C[1]),
            z_lsq_coeff_6(E2C[0]),
        )
        * z_quad_vector_sum0_6
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_7(E2C[1]),
            z_lsq_coeff_7(E2C[0]),
        )
        * z_quad_vector_sum0_7
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_8(E2C[1]),
            z_lsq_coeff_8(E2C[0]),
        )
        * z_quad_vector_sum0_8
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_9(E2C[1]),
            z_lsq_coeff_9(E2C[0]),
        )
        * z_quad_vector_sum0_9
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_10(E2C[1]),
            z_lsq_coeff_10(E2C[0]),
        )
        * z_quad_vector_sum0_10
    )

    return p_out_e_hybrid_1a


@program(grid_type=GridType.UNSTRUCTURED)
def hflux_ffsl_hybrid_stencil_01a(
    z_lsq_coeff_1: Field[[CellDim, KDim], float],
    z_lsq_coeff_2: Field[[CellDim, KDim], float],
    z_lsq_coeff_3: Field[[CellDim, KDim], float],
    z_lsq_coeff_4: Field[[CellDim, KDim], float],
    z_lsq_coeff_5: Field[[CellDim, KDim], float],
    z_lsq_coeff_6: Field[[CellDim, KDim], float],
    z_lsq_coeff_7: Field[[CellDim, KDim], float],
    z_lsq_coeff_8: Field[[CellDim, KDim], float],
    z_lsq_coeff_9: Field[[CellDim, KDim], float],
    z_lsq_coeff_10: Field[[CellDim, KDim], float],
    z_quad_vector_sum0_1: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_2: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_3: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_4: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_5: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_6: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_7: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_8: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_9: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum0_10: Field[[EdgeDim, KDim], float],
    patch0_cell_rel_idx_dsl: Field[[EdgeDim, KDim], int32],
    p_out_e_hybrid_1a: Field[[EdgeDim, KDim], float],
):
    _hflux_ffsl_hybrid_stencil_01a(
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
        out=(p_out_e_hybrid_1a),
    )
