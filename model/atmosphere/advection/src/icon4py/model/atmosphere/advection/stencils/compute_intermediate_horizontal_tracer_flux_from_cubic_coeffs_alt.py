# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_intermediate_horizontal_tracer_flux_from_cubic_coeffs_alt(
    z_lsq_coeff_1: fa.CellKField[wpfloat],
    z_lsq_coeff_2: fa.CellKField[wpfloat],
    z_lsq_coeff_3: fa.CellKField[wpfloat],
    z_lsq_coeff_4: fa.CellKField[wpfloat],
    z_lsq_coeff_5: fa.CellKField[wpfloat],
    z_lsq_coeff_6: fa.CellKField[wpfloat],
    z_lsq_coeff_7: fa.CellKField[wpfloat],
    z_lsq_coeff_8: fa.CellKField[wpfloat],
    z_lsq_coeff_9: fa.CellKField[wpfloat],
    z_lsq_coeff_10: fa.CellKField[wpfloat],
    z_quad_vector_sum_1: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_2: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_3: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_4: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_5: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_6: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_7: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_8: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_9: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_10: fa.EdgeKField[vpfloat],
    z_dreg_area: fa.EdgeKField[vpfloat],
    p_mass_flx_e: fa.EdgeKField[wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[int32],
) -> fa.EdgeKField[wpfloat]:
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
def compute_intermediate_horizontal_tracer_flux_from_cubic_coeffs_alt(
    z_lsq_coeff_1: fa.CellKField[wpfloat],
    z_lsq_coeff_2: fa.CellKField[wpfloat],
    z_lsq_coeff_3: fa.CellKField[wpfloat],
    z_lsq_coeff_4: fa.CellKField[wpfloat],
    z_lsq_coeff_5: fa.CellKField[wpfloat],
    z_lsq_coeff_6: fa.CellKField[wpfloat],
    z_lsq_coeff_7: fa.CellKField[wpfloat],
    z_lsq_coeff_8: fa.CellKField[wpfloat],
    z_lsq_coeff_9: fa.CellKField[wpfloat],
    z_lsq_coeff_10: fa.CellKField[wpfloat],
    z_quad_vector_sum_1: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_2: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_3: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_4: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_5: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_6: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_7: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_8: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_9: fa.EdgeKField[vpfloat],
    z_quad_vector_sum_10: fa.EdgeKField[vpfloat],
    z_dreg_area: fa.EdgeKField[vpfloat],
    p_mass_flx_e: fa.EdgeKField[wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[int32],
    p_out_e_miura3: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_intermediate_horizontal_tracer_flux_from_cubic_coeffs_alt(
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
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )