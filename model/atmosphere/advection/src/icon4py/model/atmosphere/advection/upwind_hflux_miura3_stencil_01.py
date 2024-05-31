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

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import where
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _upwind_hflux_miura3_stencil_01(
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
    z_quad_vector_sum_1: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_2: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_3: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_4: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_5: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_6: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_7: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_8: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_9: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_10: Field[[EdgeDim, KDim], float],
    z_dreg_area: Field[[EdgeDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    cell_rel_idx_dsl: fa.EKintField,
) -> Field[[EdgeDim, KDim], float]:
    p_out_e_miura3 = (
        (
            where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_1(E2C[1]),
                z_lsq_coeff_1(E2C[0]),
            )
            * z_quad_vector_sum_1
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_2(E2C[1]),
                z_lsq_coeff_2(E2C[0]),
            )
            * z_quad_vector_sum_2
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_3(E2C[1]),
                z_lsq_coeff_3(E2C[0]),
            )
            * z_quad_vector_sum_3
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_4(E2C[1]),
                z_lsq_coeff_4(E2C[0]),
            )
            * z_quad_vector_sum_4
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_5(E2C[1]),
                z_lsq_coeff_5(E2C[0]),
            )
            * z_quad_vector_sum_5
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_6(E2C[1]),
                z_lsq_coeff_6(E2C[0]),
            )
            * z_quad_vector_sum_6
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_7(E2C[1]),
                z_lsq_coeff_7(E2C[0]),
            )
            * z_quad_vector_sum_7
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_8(E2C[1]),
                z_lsq_coeff_8(E2C[0]),
            )
            * z_quad_vector_sum_8
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_9(E2C[1]),
                z_lsq_coeff_9(E2C[0]),
            )
            * z_quad_vector_sum_9
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_10(E2C[1]),
                z_lsq_coeff_10(E2C[0]),
            )
            * z_quad_vector_sum_10
        )
        / z_dreg_area
        * p_mass_flx_e
    )

    return p_out_e_miura3


@program(grid_type=GridType.UNSTRUCTURED)
def upwind_hflux_miura3_stencil_01(
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
    z_quad_vector_sum_1: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_2: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_3: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_4: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_5: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_6: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_7: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_8: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_9: Field[[EdgeDim, KDim], float],
    z_quad_vector_sum_10: Field[[EdgeDim, KDim], float],
    z_dreg_area: Field[[EdgeDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    cell_rel_idx_dsl: fa.EKintField,
    p_out_e_miura3: Field[[EdgeDim, KDim], float],
):
    _upwind_hflux_miura3_stencil_01(
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
        out=(p_out_e_miura3),
    )
