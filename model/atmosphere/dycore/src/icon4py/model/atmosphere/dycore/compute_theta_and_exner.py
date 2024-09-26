# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import exp, log, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_theta_and_exner(
    bdy_halo_c: fa.CellField[bool],
    rho: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_66."""
    theta_v_wp = where(bdy_halo_c, exner, theta_v)
    exner_wp = where(bdy_halo_c, exp(rd_o_cvd * log(rd_o_p0ref * rho * exner)), exner)
    return theta_v_wp, exner_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_theta_and_exner(
    bdy_halo_c: fa.CellField[bool],
    rho: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_theta_and_exner(
        bdy_halo_c,
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        out=(theta_v, exner),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
