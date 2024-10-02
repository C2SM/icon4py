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

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C


@field_operator
def _upwind_hflux_miura3_stencil_01(
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
    z_quad_vector_sum_1: fa.EdgeKField[float],
    z_quad_vector_sum_2: fa.EdgeKField[float],
    z_quad_vector_sum_3: fa.EdgeKField[float],
    z_quad_vector_sum_4: fa.EdgeKField[float],
    z_quad_vector_sum_5: fa.EdgeKField[float],
    z_quad_vector_sum_6: fa.EdgeKField[float],
    z_quad_vector_sum_7: fa.EdgeKField[float],
    z_quad_vector_sum_8: fa.EdgeKField[float],
    z_quad_vector_sum_9: fa.EdgeKField[float],
    z_quad_vector_sum_10: fa.EdgeKField[float],
    z_dreg_area: fa.EdgeKField[float],
    p_mass_flx_e: fa.EdgeKField[float],
    cell_rel_idx_dsl: fa.EdgeKField[int32],
) -> fa.EdgeKField[float]:
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
    z_quad_vector_sum_1: fa.EdgeKField[float],
    z_quad_vector_sum_2: fa.EdgeKField[float],
    z_quad_vector_sum_3: fa.EdgeKField[float],
    z_quad_vector_sum_4: fa.EdgeKField[float],
    z_quad_vector_sum_5: fa.EdgeKField[float],
    z_quad_vector_sum_6: fa.EdgeKField[float],
    z_quad_vector_sum_7: fa.EdgeKField[float],
    z_quad_vector_sum_8: fa.EdgeKField[float],
    z_quad_vector_sum_9: fa.EdgeKField[float],
    z_quad_vector_sum_10: fa.EdgeKField[float],
    z_dreg_area: fa.EdgeKField[float],
    p_mass_flx_e: fa.EdgeKField[float],
    cell_rel_idx_dsl: fa.EdgeKField[int32],
    p_out_e_miura3: fa.EdgeKField[float],
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
