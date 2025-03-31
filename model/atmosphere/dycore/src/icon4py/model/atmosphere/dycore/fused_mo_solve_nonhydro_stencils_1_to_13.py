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
from gt4py.next.ffront.fbuiltins import Field, bool, broadcast, maximum, where

from icon4py.model.atmosphere.dycore.solve_nonhydro_stencils import (
    _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures,
)
from icon4py.model.atmosphere.dycore.stencils.compute_first_vertical_derivative import (
    _compute_first_and_second_vertical_derivative_exner,
    _compute_first_vertical_derivative_igradp_method,
)
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    _compute_rho_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_temporally_exner_pressure import (
    _extrapolate_temporally_exner_pressure,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_with_zero_vp import (
    _init_two_cell_kdim_fields_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.atmosphere.dycore.stencils.set_theta_v_prime_ic_at_lower_boundary import (
    _set_theta_v_prime_ic_at_lower_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)


@field_operator
def _fused_mo_solve_nonhydro_stencils_1_to_13(
    rho_nnow: fa.CellKField[ta.wpfloat],
    rho_ref_mc: fa.CellKField[ta.wpfloat],
    theta_v_nnow: fa.CellKField[ta.wpfloat],
    theta_ref_mc: fa.CellKField[ta.wpfloat],
    z_rth_pr_1: fa.CellKField[ta.wpfloat],
    z_rth_pr_2: fa.CellKField[ta.wpfloat],
    z_theta_v_pr_ic: fa.CellKField[ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.wpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.wpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.wpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    z_exner_ic: fa.CellKField[ta.wpfloat],
    z_exner_ex_pr: fa.CellKField[ta.wpfloat],
    z_dexner_dz_c_1: fa.CellKField[ta.wpfloat],
    z_dexner_dz_c_2: fa.CellKField[ta.wpfloat],
    theta_v_ic: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[ta.wpfloat],
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
    limited_area: bool,
    igradp_method: gtx.int32,
    n_lev: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_end: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    (z_rth_pr_1, z_rth_pr_2) = (
        where(
            (start_cell_lateral_boundary <= horz_idx < end_cell_end),
            _init_two_cell_kdim_fields_with_zero_vp(),
            (z_rth_pr_1, z_rth_pr_2),
        )
        if limited_area
        else (z_rth_pr_1, z_rth_pr_2)
    )

    z_exner_ic = (
        where(
            (
                (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo)
                & (maximum(1, nflatlev) <= vert_idx)
            ),
            _interpolate_cell_field_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_exner_ex_pr),
            z_exner_ic,
        )
        if igradp_method == 3
        else z_exner_ic
    )

    (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = where(
        (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo),
        _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
            rho=rho_nnow,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v_nnow,
            theta_ref_mc=theta_ref_mc,
            rho_ic=rho_ic,
            wgtfac_c=wgtfac_c,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        ),
        (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )

    z_theta_v_pr_ic = where(
        vert_idx == 0, broadcast(0.0, (dims.CellDim, dims.KDim)), z_theta_v_pr_ic
    )

    (z_rth_pr_1, z_rth_pr_2) = where(
        (start_cell_halo_level_2 <= horz_idx < end_cell_halo_level_2),
        _compute_perturbation_of_rho_and_theta(
            rho=rho_nnow,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v_nnow,
            theta_ref_mc=theta_ref_mc,
        ),
        (z_rth_pr_1, z_rth_pr_2),
    )

    return (
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        rho_ic,
        z_exner_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
    )


@field_operator
def _fused_mo_solve_nonhydro_stencils_1_to_13_restriced(
    wgtfacq_c: fa.CellKField[ta.wpfloat],
    z_exner_ic: fa.CellKField[ta.wpfloat],
    z_exner_ex_pr: fa.CellKField[ta.wpfloat],
    horz_idx: Field[[CellDim], gtx.int32],
    igradp_method: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    end_cell_halo: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    z_exner_ex_pr = where(
        (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo),
        _init_cell_kdim_field_with_zero_wp(),
        z_exner_ex_pr,
    )

    z_exner_ic = (
        where(
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo),
            _interpolate_to_surface(wgtfacq_c=wgtfacq_c, interpolant=z_exner_ex_pr),
            z_exner_ic,
        )
        if igradp_method == 3
        else z_exner_ic
    )

    return (
        z_exner_ex_pr,
        z_exner_ic,
    )


@field_operator
def _fused_mo_solve_nonhydro_stencils_1_to_13_corrector(
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
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = where(
        (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_local) & (1 <= vert_idx),
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
    )
    return (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c)


@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_solve_nonhydro_stencils_1_to_13_predictor(
    rho_nnow: fa.CellKField[ta.wpfloat],
    rho_ref_mc: fa.CellKField[ta.wpfloat],
    theta_v_nnow: fa.CellKField[ta.wpfloat],
    theta_ref_mc: fa.CellKField[ta.wpfloat],
    z_rth_pr_1: fa.CellKField[ta.wpfloat],
    z_rth_pr_2: fa.CellKField[ta.wpfloat],
    z_theta_v_pr_ic: fa.CellKField[ta.wpfloat],
    theta_ref_ic: fa.CellKField[ta.wpfloat],
    wgtfacq_c: fa.CellKField[ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.wpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.wpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.wpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    z_exner_ic: fa.CellKField[ta.wpfloat],
    exner_exfac: fa.CellKField[ta.wpfloat],
    exner_nnow: fa.CellKField[ta.wpfloat],
    exner_ref_mc: fa.CellKField[ta.wpfloat],
    z_exner_ex_pr: fa.CellKField[ta.wpfloat],
    z_dexner_dz_c_1: fa.CellKField[ta.wpfloat],
    z_dexner_dz_c_2: fa.CellKField[ta.wpfloat],
    theta_v_ic: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[ta.wpfloat],
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
    limited_area: bool,
    igradp_method: gtx.int32,
    n_lev: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_end: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _extrapolate_temporally_exner_pressure(
        exner_exfac=exner_exfac,
        exner=exner_nnow,
        exner_ref_mc=exner_ref_mc,
        exner_pr=exner_pr,
        out=(z_exner_ex_pr, exner_pr),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _fused_mo_solve_nonhydro_stencils_1_to_13_restriced(
        wgtfacq_c=wgtfacq_c,
        z_exner_ic=z_exner_ic,
        z_exner_ex_pr=z_exner_ex_pr,
        horz_idx=horz_idx,
        igradp_method=igradp_method,
        start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
        end_cell_halo=end_cell_halo,
        out=(
            z_exner_ex_pr,
            z_exner_ic,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (n_lev, n_lev + 1),
        },
    )

    _fused_mo_solve_nonhydro_stencils_1_to_13(
        rho_nnow=rho_nnow,
        rho_ref_mc=rho_ref_mc,
        theta_v_nnow=theta_v_nnow,
        theta_ref_mc=theta_ref_mc,
        z_rth_pr_1=z_rth_pr_1,
        z_rth_pr_2=z_rth_pr_2,
        z_theta_v_pr_ic=z_theta_v_pr_ic,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        exner_pr=exner_pr,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        z_exner_ic=z_exner_ic,
        z_exner_ex_pr=z_exner_ex_pr,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        limited_area=limited_area,
        igradp_method=igradp_method,
        n_lev=n_lev,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        start_cell_lateral_boundary=start_cell_lateral_boundary,
        start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
        start_cell_halo_level_2=start_cell_halo_level_2,
        end_cell_end=end_cell_end,
        end_cell_halo=end_cell_halo,
        end_cell_halo_level_2=end_cell_halo_level_2,
        out=(
            z_rth_pr_1,
            z_rth_pr_2,
            z_exner_ex_pr,
            exner_pr,
            rho_ic,
            z_exner_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
            z_dexner_dz_c_1,
            z_dexner_dz_c_2,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _set_theta_v_prime_ic_at_lower_boundary(
        wgtfacq_c=wgtfacq_c,
        z_rth_pr=z_rth_pr_2,
        theta_ref_ic=theta_ref_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (n_lev, n_lev + 1),
        },
    )

    _interpolate_to_surface(
        wgtfacq_c=wgtfacq_c,
        interpolant=z_exner_ex_pr,
        out=z_exner_ic,
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (n_lev, n_lev + 1),
        },
    )

    _compute_first_vertical_derivative_igradp_method(
        z_exner_ic=z_exner_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        igradp_method=igradp_method,
        out=z_dexner_dz_c_1,
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (nflatlev, vertical_end - 1),
        },
    )

    _compute_first_and_second_vertical_derivative_exner(
        z_exner_ic=z_exner_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        z_theta_v_pr_ic=z_theta_v_pr_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        z_rth_pr_2=z_rth_pr_2,
        igradp_method=igradp_method,
        nflatlev=nflatlev,
        vert_idx=vert_idx,
        nflat_gradp=nflat_gradp,
        out=(z_dexner_dz_c_1, z_dexner_dz_c_2),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )


@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_solve_nonhydro_stencils_1_to_13_corrector(
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
    _fused_mo_solve_nonhydro_stencils_1_to_13_corrector(
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
