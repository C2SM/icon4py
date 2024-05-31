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
def _upwind_hflux_miura_cycl_stencil_01(
    z_lsq_coeff_1_dsl: Field[[CellDim, KDim], float],
    z_lsq_coeff_2_dsl: Field[[CellDim, KDim], float],
    z_lsq_coeff_3_dsl: Field[[CellDim, KDim], float],
    distv_bary_1: Field[[EdgeDim, KDim], float],
    distv_bary_2: Field[[EdgeDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    cell_rel_idx_dsl: fa.EKintField,
) -> Field[[EdgeDim, KDim], float]:
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
    z_lsq_coeff_1_dsl: Field[[CellDim, KDim], float],
    z_lsq_coeff_2_dsl: Field[[CellDim, KDim], float],
    z_lsq_coeff_3_dsl: Field[[CellDim, KDim], float],
    distv_bary_1: Field[[EdgeDim, KDim], float],
    distv_bary_2: Field[[EdgeDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    cell_rel_idx_dsl: fa.EKintField,
    z_tracer_mflx_dsl: Field[[EdgeDim, KDim], float],
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
