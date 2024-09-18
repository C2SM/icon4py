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


# TODO (dastrm): this stencil has no test


@field_operator
def _compute_intermediate_horizontal_tracer_flux_from_cubic_coefficients(
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
    z_quad_vector_sum0_1: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_2: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_3: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_4: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_5: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_6: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_7: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_8: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_9: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_10: fa.EdgeKField[vpfloat],
    patch0_cell_rel_idx_dsl: fa.EdgeKField[int32],
) -> fa.EdgeKField[wpfloat]:
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
def compute_intermediate_horizontal_tracer_flux_from_cubic_coefficients(
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
    z_quad_vector_sum0_1: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_2: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_3: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_4: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_5: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_6: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_7: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_8: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_9: fa.EdgeKField[vpfloat],
    z_quad_vector_sum0_10: fa.EdgeKField[vpfloat],
    patch0_cell_rel_idx_dsl: fa.EdgeKField[int32],
    p_out_e_hybrid_1a: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_intermediate_horizontal_tracer_flux_from_cubic_coefficients(
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
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
