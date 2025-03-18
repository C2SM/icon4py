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
    _predictor_stencils_11_lower_upper,
)
from icon4py.model.atmosphere.dycore.stencils.compute_approx_of_2nd_vertical_derivative_of_exner import (
    _compute_approx_of_2nd_vertical_derivative_of_exner,
)
from icon4py.model.atmosphere.dycore.stencils.compute_first_vertical_derivative import (
    _compute_first_vertical_derivative,
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
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import CellDim, KDim

@field_operator
def _fused_mo_solve_nonhydro_stencils_1_to_5(
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
    limited_area: bool,
    igradp_method: gtx.int32,
    n_lev: gtx.int32,
    nflatlev: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    end_cell_end: gtx.int32,
    end_cell_halo: gtx.int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    vert_idx_1d = vert_idx
    vert_start = 0
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    # stencil_01
    (z_rth_pr_1, z_rth_pr_2) = (
        where(
            (start_cell_lateral_boundary <= horz_idx < end_cell_end),
            _init_two_cell_kdim_fields_with_zero_vp(),
            (z_rth_pr_1, z_rth_pr_2),
        )
        if limited_area
        else (z_rth_pr_1, z_rth_pr_2)
    )

    # solve_nonhydro_stencil_02_03
    (z_exner_ex_pr, exner_pr) = where(
        ((start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx < n_lev)),
        _extrapolate_temporally_exner_pressure(
            exner_exfac=exner_exfac,
            exner=exner_nnow,
            exner_ref_mc=exner_ref_mc,
            exner_pr=exner_pr,
        ),
        (z_exner_ex_pr, exner_pr),
    )

    z_exner_ex_pr = where(
        ((start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx == n_lev)),
        _init_cell_kdim_field_with_zero_wp(),
        z_exner_ex_pr,
    )

    vert_start = maximum(1, nflatlev) if igradp_method == 3 else vert_start

    z_exner_ic = (
        where(
            (
                (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo)
                & (vert_idx == n_lev)
            ),
            _interpolate_to_surface(
                wgtfacq_c=wgtfacq_c_dsl, interpolant=z_exner_ex_pr
            ), z_exner_ic
        )
        if igradp_method == 3
        else z_exner_ic
    )

    z_exner_ic = (
        where(
            (
                (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo)
                & (vert_start <= vert_idx < n_lev)
            ),
            _interpolate_to_half_levels_vp(
                wgtfac_c=wgtfac_c, interpolant=z_exner_ex_pr
            ), z_exner_ic
        )
        if igradp_method == 3
        else z_exner_ic
    )

    return (
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
    )


@field_operator
def _fused_mo_solve_nonhydro_stencils_6_to_13(
    rho_nnow: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], gtx.int32],
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
    igradp_method: gtx.int32,
    n_lev: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    vert_idx_1d = vert_idx
    vert_start = 0
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    vert_start = maximum(1, nflatlev) if igradp_method == 3 else vert_start

    z_dexner_dz_c_1 = (
        where(
            ((start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo)& (vert_start <= vert_idx < n_lev)),
            _compute_first_vertical_derivative(
                z_exner_ic=z_exner_ic,
                inv_ddqz_z_full=inv_ddqz_z_full
            ),z_dexner_dz_c_1)  if igradp_method == 3 else z_dexner_dz_c_1
    )

    # solve_nonhydro_stencil_7_8_9, which is already combined
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
            k_field=k_field,
        ),
        (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )

    # stencil_11
    (z_theta_v_pr_ic, theta_v_ic) = where(
        ((start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx <= n_lev)),
        _predictor_stencils_11_lower_upper(
            wgtfacq_c_dsl=wgtfacq_c_dsl,
            z_rth_pr=z_rth_pr_2,
            theta_ref_ic=theta_ref_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            k_field=k_field,
            nlev=n_lev,
        ),
        (z_theta_v_pr_ic, theta_v_ic),
    )

    # stencil 12
    vert_start = nflat_gradp if igradp_method == 3 else vert_start

    z_dexner_dz_c_1 = where(
            ((start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_start <= vert_idx < n_lev)),
            _compute_first_vertical_derivative(
                z_exner_ic=z_exner_ic,
                inv_ddqz_z_full=inv_ddqz_z_full,
            ),z_dexner_dz_c_1)  if igradp_method == 3 else z_dexner_dz_c_1

    # stencil 13
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
def _fused_mo_solve_nonhydro_stencils_1_to_13_corrector(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
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
    rho_nnow: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], gtx.int32],
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    limited_area: bool,
    igradp_method: gtx.int32,
    n_lev: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_end: gtx.int32,
    end_cell_local: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    _fused_mo_solve_nonhydro_stencils_1_to_5(
        z_rth_pr_1,
        z_rth_pr_2,
        wgtfacq_c_dsl,
        wgtfac_c,
        exner_pr,
        z_exner_ic,
        exner_exfac,
        exner_nnow,
        exner_ref_mc,
        z_exner_ex_pr,
        horz_idx,
        vert_idx,
        limited_area,
        igradp_method,
        n_lev,
        nflatlev,
        start_cell_lateral_boundary,
        start_cell_lateral_boundary_level_3,
        end_cell_end,
        end_cell_halo,
        out=(
            z_rth_pr_1,
            z_rth_pr_2,
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _fused_mo_solve_nonhydro_stencils_1_to_5(
        z_rth_pr_1,
        z_rth_pr_2,
        wgtfacq_c_dsl,
        wgtfac_c,
        exner_pr,
        z_exner_ic,
        exner_exfac,
        exner_nnow,
        exner_ref_mc,
        z_exner_ex_pr,
        horz_idx,
        vert_idx,
        limited_area,
        igradp_method,
        n_lev,
        nflatlev,
        start_cell_lateral_boundary,
        start_cell_lateral_boundary_level_3,
        end_cell_end,
        end_cell_halo,
        out=(
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
    ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end-1, vertical_end),
        },
    )

    _fused_mo_solve_nonhydro_stencils_6_to_13(
        rho_nnow,
        rho_ref_mc,
        theta_v_nnow,
        theta_ref_mc,
        z_rth_pr_1,
        z_rth_pr_2,
        z_theta_v_pr_ic,
        theta_ref_ic,
        wgtfacq_c_dsl,
        wgtfac_c,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_th_ddz_exner_c,
        k_field,
        rho_ic,
        z_exner_ic,
        z_exner_ex_pr,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        theta_v_ic,
        inv_ddqz_z_full,
        horz_idx,
        vert_idx,
        igradp_method,
        n_lev,
        nflatlev,
        nflat_gradp,
        start_cell_lateral_boundary_level_3,
        start_cell_halo_level_2,
        end_cell_halo,
        end_cell_halo_level_2,
        out=(z_rth_pr_1,
             z_rth_pr_2,
             z_exner_ex_pr,
             exner_pr,
             rho_ic,
             z_exner_ic,
             z_theta_v_pr_ic,
             theta_v_ic,
             z_th_ddz_exner_c,
             z_dexner_dz_c_1,
             z_dexner_dz_c_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _fused_mo_solve_nonhydro_stencils_6_to_13(
        rho_nnow,
        rho_ref_mc,
        theta_v_nnow,
        theta_ref_mc,
        z_rth_pr_1,
        z_rth_pr_2,
        z_theta_v_pr_ic,
        theta_ref_ic,
        wgtfacq_c_dsl,
        wgtfac_c,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_th_ddz_exner_c,
        k_field,
        rho_ic,
        z_exner_ic,
        z_exner_ex_pr,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        theta_v_ic,
        inv_ddqz_z_full,
        horz_idx,
        vert_idx,
        igradp_method,
        n_lev,
        nflatlev,
        nflat_gradp,
        start_cell_lateral_boundary_level_3,
        start_cell_halo_level_2,
        end_cell_halo,
        end_cell_halo_level_2,
        out=(z_rth_pr_1,
             z_rth_pr_2,
             z_exner_ex_pr,
             exner_pr,
             rho_ic,
             z_exner_ic,
             z_theta_v_pr_ic,
             theta_v_ic,
             z_th_ddz_exner_c,
             z_dexner_dz_c_1,
             z_dexner_dz_c_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end-1, vertical_end),
        },
    )

@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_solve_nonhydro_stencils_1_to_13_corrector(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    horz_idx: Field[[CellDim], gtx.int32],
    vert_idx: Field[[KDim], gtx.int32],
    start_cell_lateral_boundary_level_3: gtx.int32,
    end_cell_local: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
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
        out=(rho_ic,
             z_theta_v_pr_ic,
             theta_v_ic,
             z_th_ddz_exner_c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end-1),
        },
    )

# TODO: check the size in serialbox.py
