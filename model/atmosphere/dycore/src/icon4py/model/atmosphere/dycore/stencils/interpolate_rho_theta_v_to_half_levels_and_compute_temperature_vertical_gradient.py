# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

import gt4py.next as gtx
from gt4py.next import program
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.atmosphere.dycore.stencils.compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    _compute_rho_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _interpolate_rho_theta_v_to_half_levels_and_compute_temperature_vertical_gradient(
    w: fa.CellKField[ta.vpfloat],
    w_concorr_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    rho_nnow: fa.CellKField[ta.vpfloat],
    rho_nvar: fa.CellKField[ta.vpfloat],
    theta_v_nnow: fa.CellKField[ta.vpfloat],
    theta_v_nvar: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.wpfloat],
    theta_ref_mc: fa.CellKField[ta.wpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.wpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    z_theta_v_pr_ic: fa.CellKField[ta.wpfloat],
    theta_v_ic: fa.CellKField[ta.wpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
    wgt_nnow_rth: ta.wpfloat,
    wgt_nnew_rth: ta.wpfloat,
    horz_idx: fa.CellField[gtx.int32],
    vert_idx: fa.KField[gtx.int32],
    start_cell_lateral_boundary_level_3: gtx.int32,
    end_cell_local: gtx.int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = concat_where(
        1 <= dims.KDim,
        concat_where(
            (start_cell_lateral_boundary_level_3 <= dims.CellDim) & (dims.CellDim < end_cell_local),
            _compute_rho_virtual_potential_temperatures_and_pressure_gradient(
                w=w,
                w_concorr_c=w_concorr_c,
                ddqz_z_half=ddqz_z_half,
                rho_now=rho_nnow,
                rho_var=rho_nvar,
                theta_now=theta_v_nnow,
                theta_var=theta_v_nvar,
                wgtfac_c=wgtfac_c,
                theta_ref_mc=theta_ref_mc,
                vwind_expl_wgt=vwind_expl_wgt,
                exner_pr=exner_pr,
                d_exner_dz_ref_ic=d_exner_dz_ref_ic,
                dtime=dtime,
                wgt_nnow_rth=wgt_nnow_rth,
                wgt_nnew_rth=wgt_nnew_rth,
            ),
            (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
        ),
        (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )
    return (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c)


@program(grid_type=GridType.UNSTRUCTURED)
def interpolate_rho_theta_v_to_half_levels_and_compute_temperature_vertical_gradient(
    w: fa.CellKField[ta.wpfloat],
    w_concorr_c: fa.CellKField[ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.wpfloat],
    rho_nnow: fa.CellKField[ta.wpfloat],
    rho_nvar: fa.CellKField[ta.wpfloat],
    theta_v_nnow: fa.CellKField[ta.wpfloat],
    theta_v_nvar: fa.CellKField[ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.wpfloat],
    theta_ref_mc: fa.CellKField[ta.wpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.wpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    z_theta_v_pr_ic: fa.CellKField[ta.wpfloat],
    theta_v_ic: fa.CellKField[ta.wpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
    wgt_nnow_rth: ta.wpfloat,
    wgt_nnew_rth: ta.wpfloat,
    horz_idx: fa.CellField[gtx.int32],
    vert_idx: fa.KField[gtx.int32],
    start_cell_lateral_boundary_level_3: gtx.int32,
    end_cell_local: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_rho_theta_v_to_half_levels_and_compute_temperature_vertical_gradient(
        w,
        w_concorr_c,
        ddqz_z_half,
        rho_nnow,
        rho_nvar,
        theta_v_nnow,
        theta_v_nvar,
        wgtfac_c,
        theta_ref_mc,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        rho_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        horz_idx,
        vert_idx,
        start_cell_lateral_boundary_level_3,
        end_cell_local,
        out=(rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
