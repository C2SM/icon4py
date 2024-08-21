# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _add_analysis_increments_from_data_assimilation(
    z_rho_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    rho_incr: fa.CellKField[vpfloat],
    exner_incr: fa.CellKField[vpfloat],
    iau_wgt_dyn: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_50."""
    rho_incr_wp, exner_incr_wp = astype((rho_incr, exner_incr), wpfloat)

    z_rho_expl_wp = z_rho_expl + iau_wgt_dyn * rho_incr_wp
    z_exner_expl_wp = z_exner_expl + iau_wgt_dyn * exner_incr_wp
    return z_rho_expl_wp, z_exner_expl_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_analysis_increments_from_data_assimilation(
    z_rho_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    rho_incr: fa.CellKField[vpfloat],
    exner_incr: fa.CellKField[vpfloat],
    iau_wgt_dyn: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _add_analysis_increments_from_data_assimilation(
        z_rho_expl,
        z_exner_expl,
        rho_incr,
        exner_incr,
        iau_wgt_dyn,
        out=(z_rho_expl, z_exner_expl),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
