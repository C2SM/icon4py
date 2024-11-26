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

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff


@field_operator
def _mo_solve_nonhydro_stencil_51(
    vwind_impl_wgt: fa.CellField[float],
    theta_v_ic: fa.CellKField[float],
    ddqz_z_half: fa.CellKField[float],
    z_beta: fa.CellKField[float],
    z_alpha: fa.CellKField[float],
    z_w_expl: fa.CellKField[float],
    z_exner_expl: fa.CellKField[float],
    dtime: float,
    cpd: float,
) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_c = -z_gamma * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q = -z_c / z_b
    w_nnew = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    w_nnew = w_nnew / z_b

    return z_q, w_nnew


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_51(
    z_q: fa.CellKField[float],
    w_nnew: fa.CellKField[float],
    vwind_impl_wgt: fa.CellField[float],
    theta_v_ic: fa.CellKField[float],
    ddqz_z_half: fa.CellKField[float],
    z_beta: fa.CellKField[float],
    z_alpha: fa.CellKField[float],
    z_w_expl: fa.CellKField[float],
    z_exner_expl: fa.CellKField[float],
    dtime: float,
    cpd: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _mo_solve_nonhydro_stencil_51(
        vwind_impl_wgt,
        theta_v_ic,
        ddqz_z_half,
        z_beta,
        z_alpha,
        z_w_expl,
        z_exner_expl,
        dtime,
        cpd,
        out=(z_q, w_nnew),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
