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
def _upwind_hflux_miura_cycl_stencil_01(
    z_lsq_coeff_1_dsl: fa.CellKField[float],
    z_lsq_coeff_2_dsl: fa.CellKField[float],
    z_lsq_coeff_3_dsl: fa.CellKField[float],
    distv_bary_1: fa.EdgeKField[float],
    distv_bary_2: fa.EdgeKField[float],
    p_mass_flx_e: fa.EdgeKField[float],
    cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
) -> fa.EdgeKField[float]:
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
def upwind_hflux_miura_cycl_stencil_01(
    z_lsq_coeff_1_dsl: fa.CellKField[float],
    z_lsq_coeff_2_dsl: fa.CellKField[float],
    z_lsq_coeff_3_dsl: fa.CellKField[float],
    distv_bary_1: fa.EdgeKField[float],
    distv_bary_2: fa.EdgeKField[float],
    p_mass_flx_e: fa.EdgeKField[float],
    cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    z_tracer_mflx_dsl: fa.EdgeKField[float],
):
    _upwind_hflux_miura_cycl_stencil_01(
        z_lsq_coeff_1_dsl,
        z_lsq_coeff_2_dsl,
        z_lsq_coeff_3_dsl,
        distv_bary_1,
        distv_bary_2,
        p_mass_flx_e,
        cell_rel_idx_dsl,
        out=(z_tracer_mflx_dsl),
    )
