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
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _update_dynamical_exner_time_increment(
    exner: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    exner_dyn_incr: fa.CellKField[vpfloat],
    ndyn_substeps_var: wpfloat,
    dtime: wpfloat,
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_60."""
    exner_dyn_incr_wp, ddt_exner_phy_wp = astype((exner_dyn_incr, ddt_exner_phy), wpfloat)

    exner_dyn_incr_wp = exner - (exner_dyn_incr_wp + ndyn_substeps_var * dtime * ddt_exner_phy_wp)
    return astype(exner_dyn_incr_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def update_dynamical_exner_time_increment(
    exner: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    exner_dyn_incr: fa.CellKField[vpfloat],
    ndyn_substeps_var: wpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_dynamical_exner_time_increment(
        exner,
        ddt_exner_phy,
        exner_dyn_incr,
        ndyn_substeps_var,
        dtime,
        out=exner_dyn_incr,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
