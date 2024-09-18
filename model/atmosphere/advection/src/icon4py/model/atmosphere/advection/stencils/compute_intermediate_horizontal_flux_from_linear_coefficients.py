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
def _compute_intermediate_horizontal_flux_from_linear_coefficients(
    z_lsq_coeff_1_dsl: fa.CellKField[wpfloat],
    z_lsq_coeff_2_dsl: fa.CellKField[wpfloat],
    z_lsq_coeff_3_dsl: fa.CellKField[wpfloat],
    distv_bary_1: fa.EdgeKField[vpfloat],
    distv_bary_2: fa.EdgeKField[vpfloat],
    p_mass_flx_e: fa.EdgeKField[wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[int32],
) -> fa.EdgeKField[wpfloat]:
    z_tracer_mflx_dsl = (
        where(
            cell_rel_idx_dsl == 1,
            z_lsq_coeff_1_dsl(E2C[1]),
            z_lsq_coeff_1_dsl(E2C[0]),
        )
        + distv_bary_1
        * where(
            cell_rel_idx_dsl == 1,
            z_lsq_coeff_2_dsl(E2C[1]),
            z_lsq_coeff_2_dsl(E2C[0]),
        )
        + distv_bary_2
        * where(
            cell_rel_idx_dsl == 1,
            z_lsq_coeff_3_dsl(E2C[1]),
            z_lsq_coeff_3_dsl(E2C[0]),
        )
    ) * p_mass_flx_e

    return z_tracer_mflx_dsl


@program(grid_type=GridType.UNSTRUCTURED)
def compute_intermediate_horizontal_flux_from_linear_coefficients(
    z_lsq_coeff_1_dsl: fa.CellKField[wpfloat],
    z_lsq_coeff_2_dsl: fa.CellKField[wpfloat],
    z_lsq_coeff_3_dsl: fa.CellKField[wpfloat],
    distv_bary_1: fa.EdgeKField[vpfloat],
    distv_bary_2: fa.EdgeKField[vpfloat],
    p_mass_flx_e: fa.EdgeKField[wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[int32],
    z_tracer_mflx_dsl: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_intermediate_horizontal_flux_from_linear_coefficients(
        z_lsq_coeff_1_dsl,
        z_lsq_coeff_2_dsl,
        z_lsq_coeff_3_dsl,
        distv_bary_1,
        distv_bary_2,
        p_mass_flx_e,
        cell_rel_idx_dsl,
        out=(z_tracer_mflx_dsl),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
