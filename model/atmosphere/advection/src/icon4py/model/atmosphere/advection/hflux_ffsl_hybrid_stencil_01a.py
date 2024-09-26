# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C


@field_operator
def _hflux_ffsl_hybrid_stencil_01a(
    z_lsq_coeff_1: fa.CellKField[float],
    z_lsq_coeff_2: fa.CellKField[float],
    z_lsq_coeff_3: fa.CellKField[float],
    z_lsq_coeff_4: fa.CellKField[float],
    z_lsq_coeff_5: fa.CellKField[float],
    z_lsq_coeff_6: fa.CellKField[float],
    z_lsq_coeff_7: fa.CellKField[float],
    z_lsq_coeff_8: fa.CellKField[float],
    z_lsq_coeff_9: fa.CellKField[float],
    z_lsq_coeff_10: fa.CellKField[float],
    z_quad_vector_sum0_1: fa.EdgeKField[float],
    z_quad_vector_sum0_2: fa.EdgeKField[float],
    z_quad_vector_sum0_3: fa.EdgeKField[float],
    z_quad_vector_sum0_4: fa.EdgeKField[float],
    z_quad_vector_sum0_5: fa.EdgeKField[float],
    z_quad_vector_sum0_6: fa.EdgeKField[float],
    z_quad_vector_sum0_7: fa.EdgeKField[float],
    z_quad_vector_sum0_8: fa.EdgeKField[float],
    z_quad_vector_sum0_9: fa.EdgeKField[float],
    z_quad_vector_sum0_10: fa.EdgeKField[float],
    patch0_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
) -> fa.EdgeKField[float]:
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
    z_lsq_coeff_1: fa.CellKField[float],
    z_lsq_coeff_2: fa.CellKField[float],
    z_lsq_coeff_3: fa.CellKField[float],
    z_lsq_coeff_4: fa.CellKField[float],
    z_lsq_coeff_5: fa.CellKField[float],
    z_lsq_coeff_6: fa.CellKField[float],
    z_lsq_coeff_7: fa.CellKField[float],
    z_lsq_coeff_8: fa.CellKField[float],
    z_lsq_coeff_9: fa.CellKField[float],
    z_lsq_coeff_10: fa.CellKField[float],
    z_quad_vector_sum0_1: fa.EdgeKField[float],
    z_quad_vector_sum0_2: fa.EdgeKField[float],
    z_quad_vector_sum0_3: fa.EdgeKField[float],
    z_quad_vector_sum0_4: fa.EdgeKField[float],
    z_quad_vector_sum0_5: fa.EdgeKField[float],
    z_quad_vector_sum0_6: fa.EdgeKField[float],
    z_quad_vector_sum0_7: fa.EdgeKField[float],
    z_quad_vector_sum0_8: fa.EdgeKField[float],
    z_quad_vector_sum0_9: fa.EdgeKField[float],
    z_quad_vector_sum0_10: fa.EdgeKField[float],
    patch0_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_out_e_hybrid_1a: fa.EdgeKField[float],
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
