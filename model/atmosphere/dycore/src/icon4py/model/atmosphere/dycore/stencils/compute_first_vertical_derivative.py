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

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat

from icon4py.model.atmosphere.dycore.stencils.compute_approx_of_2nd_vertical_derivative_of_exner import (
    _compute_approx_of_2nd_vertical_derivative_of_exner,
)


@field_operator
def _compute_first_vertical_derivative(
    z_exner_ic: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_06."""
    z_dexner_dz_c_1 = (z_exner_ic - z_exner_ic(Koff[1])) * inv_ddqz_z_full
    return z_dexner_dz_c_1


@field_operator
def _compute_first_vertical_derivative_igradp_method(
    z_exner_ic: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    igradp_method: gtx.int32,
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_06."""
    z_dexner_dz_c_1 = (
        (z_exner_ic - z_exner_ic(Koff[1])) * inv_ddqz_z_full
        if igradp_method == 3
        else z_dexner_dz_c_1
    )
    return z_dexner_dz_c_1

@field_operator
def _compute_first_and_second_vertical_derivative_exner(
    z_exner_ic: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    vert_idx:fa.KField[gtx.int32],
    nflat_gradp: gtx.int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:

    z_dexner_dz_c_1 = (
        where(
            (nflatlev <= vert_idx),
        _compute_first_vertical_derivative_igradp_method(
                    z_exner_ic=z_exner_ic,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    z_dexner_dz_c_1=z_dexner_dz_c_1,
                    igradp_method=igradp_method), z_dexner_dz_c_1
        )
    )

    z_dexner_dz_c_2 = (
        where(
            (nflat_gradp <= vert_idx),
            _compute_approx_of_2nd_vertical_derivative_of_exner(
                z_theta_v_pr_ic=z_theta_v_pr_ic,
                d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
                z_rth_pr_2=z_rth_pr_2,
            ),
            z_dexner_dz_c_2,
        )
        if igradp_method == 3
        else z_dexner_dz_c_2
    )

    return z_dexner_dz_c_1, z_dexner_dz_c_2



@program(grid_type=GridType.UNSTRUCTURED)
def compute_first_vertical_derivative(
    z_exner_ic: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_first_vertical_derivative(
        z_exner_ic,
        inv_ddqz_z_full,
        out=z_dexner_dz_c_1,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
