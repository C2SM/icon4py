# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp, log, where
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_theta_and_exner(
    mask_prog_halo_c: fa.CellField[bool],
    rho: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
    start_cell_halo: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_66."""
    theta_v_wp = concat_where(
        dims.CellDim < start_cell_halo,
        where(mask_prog_halo_c, exner, theta_v),
        where(
            ~mask_prog_halo_c, exner, theta_v
        ),  # mask_prog_halo_c is the inverse of bdy_halo_c **only in the halo region**
    )
    exner_wp = concat_where(
        dims.CellDim < start_cell_halo,
        where(mask_prog_halo_c, exp(rd_o_cvd * log(rd_o_p0ref * rho * exner)), exner),
        where(
            ~mask_prog_halo_c, exp(rd_o_cvd * log(rd_o_p0ref * rho * exner)), exner
        ),  # mask_prog_halo_c is the inverse of bdy_halo_c **only in the halo region**
    )
    return theta_v_wp, exner_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_theta_and_exner(
    mask_prog_halo_c: fa.CellField[bool],
    rho: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
    start_cell_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_theta_and_exner(
        mask_prog_halo_c,
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        start_cell_halo,
        out=(theta_v, exner),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
