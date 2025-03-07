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
from gt4py.next.common import GridType
from gt4py.next.embedded.context import offset_provider
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, bool, broadcast, int32, maximum, where

from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_with_zero_vp import (
    _init_two_cell_kdim_fields_with_zero_vp
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_temporally_exner_pressure import (
    _extrapolate_temporally_exner_pressure
)

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp
)

from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import (
    _interpolate_to_surface
)

from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp
)

from icon4py.model.atmosphere.dycore.stencils.compute_first_vertical_derivative import (
    _compute_first_vertical_derivative
)

from icon4py.model.atmosphere.dycore.solve_nonhydro_stencils import (
    _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures,
    _predictor_stencils_11_lower_upper
)

from icon4py.model.atmosphere.dycore.stencils.compute_approx_of_2nd_vertical_derivative_of_exner import (
    _compute_approx_of_2nd_vertical_derivative_of_exner
)

from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta
)

from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _fused_mo_solve_nonhydro_stencils_1_to_13_predictor(
    rho_nnow: Field[[CellDim, KDim], float],
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
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_end: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32
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
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    # stencil_01
    (z_rth_pr_1, z_rth_pr_2) = where(
        (start_cell_lateral_boundary <= horz_idx < end_cell_end),
        _init_two_cell_kdim_fields_with_zero_vp(
            z_rth_pr_1, z_rth_pr_2
        ),
        (z_rth_pr_1, z_rth_pr_2)
        ) if limited_area else (z_rth_pr_1, z_rth_pr_2)

    # solve_nonhydro_stencil_02_03
    # TODO: there is a way to simplify the horizontal condition since 2_3 and 4_5_6 are sharing the same horizontal condition
    (z_exner_ex_pr, exner_pr) = where(
        (
                 (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx < n_lev)
        ),
        _extrapolate_temporally_exner_pressure(
            exner_exfac=exner_exfac,
            exner=exner_nnow,
            exner_ref_mc=exner_ref_mc,
            exner_pr=exner_pr,
            z_exner_ex_pr= z_exner_ex_pr,
            exner_pr=exner_pr
        ),
        (z_exner_ex_pr, exner_pr),
    )

    z_exner_ex_pr = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx == n_lev)
        ),
        _init_cell_kdim_field_with_zero_wp(
            exner_pr=exner_pr
        ),
        z_exner_ex_pr,
    )

    vert_start = maximum(1, nflatlev) if igradp_method ==3 else vert_start

    # solve_nonhydro_stencil_4_5_6
    z_exner_ic = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx == n_lev )
        ),
        _interpolate_to_surface(
            wgtfacq_c_dsl=wgtfacq_c_dsl,
            z_exner_ex_pr=z_exner_ex_pr,
            z_exner_ic=z_exner_ic
        )
    ) if igradp_method == 3 else z_exner_ic

    z_exner_ic = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_start <= vert_idx < n_lev)
        ),
        _interpolate_to_half_levels_vp(
            wgtfac_c=wgtfac_c,
            z_exner_ex_pr=z_exner_ex_pr,
            z_exner_ic=z_exner_ic
        )
    ) if igradp_method == 3 else z_exner_ic

    z_dexner_dz_c_1 = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_start <= vert_idx < n_lev)
        ),
        _compute_first_vertical_derivative(
            z_exner_ic=z_exner_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_dexner_dz_c_1=z_dexner_dz_c_1
        )
    ) if igradp_method == 3 else z_dexner_dz_c_1

    # solve_nonhydro_stencil_7_8_9, which is already combined
    (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo),
        ),
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
            d_exner_dz_ref_ic= d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            k_field=k_field
        ),
        (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c)
    )

    #stencil_11
    (z_theta_v_pr_ic, theta_v_ic) = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_idx <= n_lev)
        ),
        _predictor_stencils_11_lower_upper(
            wgtfacq_c_dsl= wgtfacq_c_dsl,
            z_rth_pr=z_rth_pr_2,
            theta_ref_ic=theta_ref_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            k_field=k_field,
            nlev=n_lev
        ),
        (z_theta_v_pr_ic, theta_v_ic)
    )

    # stencil 12
    vert_start = nflat_gradp if igradp_method ==3 else vert_start

    z_dexner_dz_c_2 = where(
        (
            (start_cell_lateral_boundary_level_3 <= horz_idx < end_cell_halo) & (vert_start <= vert_idx)
        ),
        _compute_approx_of_2nd_vertical_derivative_of_exner(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            z_rth_pr_2=z_rth_pr_2
        )
    ) if igradp_method ==3 else z_dexner_dz_c_2

    # stencil 13
    (z_rth_pr_1, z_rth_pr_2) = where(
        (
            (start_cell_halo_level_2 <= horz_idx < end_cell_halo_level_2),
            _compute_perturbation_of_rho_and_theta(
                rho=rho_nnow,
                rho_ref_mc=rho_ref_mc,
                theta_v=theta_v_nnow,
                theta_ref_mc=theta_ref_mc,
                z_rth_pr_1=z_rth_pr_1,
                z_rth_pr_2=z_rth_pr_2
            ),
            (z_rth_pr_1, z_rth_pr_2),
        )
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
        z_dexner_dz_c_2
    )

#TODO: how can I check what are the output of the combined stencils?


# @field_operator
# def _fused_mo_solve_nonhydro_stencils_01_to_13_corrector(
#     w: Field[[CellDim, KDim], float],
#     w_concorr_c: Field[[CellDim, KDim], float],
#     ddqz_z_half: Field[[CellDim, KDim], float],
#     rho_nnow: Field[[CellDim, KDim], float],
#     rho_nvar: Field[[CellDim, KDim], float],
#     theta_v_nnow: Field[[CellDim, KDim], float],
#     theta_v_nvar: Field[[CellDim, KDim], float],
#     wgtfac_c: Field[[CellDim, KDim], float],
#     theta_ref_mc: Field[[CellDim, KDim], float],
#     vwind_expl_wgt: Field[[CellDim], float],
#     exner_pr: Field[[CellDim, KDim], float],
#     d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
#     rho_ic: Field[[CellDim, KDim], float],
#     z_theta_v_pr_ic: Field[[CellDim, KDim], float],
#     theta_v_ic: Field[[CellDim, KDim], float],
#     z_th_ddz_exner_c: Field[[CellDim, KDim], float],
#     dtime: float,
#     wgt_nnow_rth: float,
#     wgt_nnew_rth: float,
#     horz_idx: Field[[CellDim], int32],
#     vert_idx: Field[[KDim], int32],
#     horizontal_lower_11: int32,
#     horizontal_upper_11: int32,
#     n_lev: int32,
# ) -> tuple[
#     Field[[CellDim, KDim], float],
#     Field[[CellDim, KDim], float],
#     Field[[CellDim, KDim], float],
#     Field[[CellDim, KDim], float],
# ]:
#     vert_idx = broadcast(vert_idx, (CellDim, KDim))
#
#     (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c,) = where(
#         (horizontal_lower_11 <= horz_idx < horizontal_upper_11) & (int32(1) <= vert_idx),
#         _mo_solve_nonhydro_stencil_10(
#             w=w,
#             w_concorr_c=w_concorr_c,
#             ddqz_z_half=ddqz_z_half,
#             rho_now=rho_nnow,
#             rho_var=rho_nvar,
#             theta_now=theta_v_nnow,
#             theta_var=theta_v_nvar,
#             wgtfac_c=wgtfac_c,
#             theta_ref_mc=theta_ref_mc,
#             vwind_expl_wgt=vwind_expl_wgt,
#             exner_pr=exner_pr,
#             d_exner_dz_ref_ic=d_exner_dz_ref_ic,
#             dtime=dtime,
#             wgt_nnow_rth=wgt_nnow_rth,
#             wgt_nnew_rth=wgt_nnew_rth,
#         ),
#         (
#             rho_ic,
#             z_theta_v_pr_ic,
#             theta_v_ic,
#             z_th_ddz_exner_c,
#         ),
#     )
#
#     return (
#         rho_ic,
#         z_theta_v_pr_ic,
#         theta_v_ic,
#         z_th_ddz_exner_c,
#     )


@field_operator
def _fused_mo_solve_nonhydro_stencils_01_to_13_restricted(
    rho_nnow: Field[[CellDim, KDim], float],
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
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    istep: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    horizontal_lower_11: int32,
    horizontal_upper_11: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    # if istep == 1:

    (
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
        z_dexner_dz_c_1,
    ) = (
        _fused_mo_solve_nonhydro_stencils_01_to_13_predictor(
            rho_nnow,
            rho_ref_mc,
            theta_v_nnow,
            theta_ref_mc,
            z_rth_pr_1,
            z_rth_pr_2,
            z_theta_v_pr_ic,
            theta_ref_ic,
            d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc,
            wgtfacq_c_dsl,
            wgtfac_c,
            vwind_expl_wgt,
            exner_pr,
            d_exner_dz_ref_ic,
            ddqz_z_half,
            z_th_ddz_exner_c,
            rho_ic,
            z_exner_ic,
            exner_exfac,
            exner_nnow,
            exner_ref_mc,
            z_exner_ex_pr,
            z_dexner_dz_c_1,
            z_dexner_dz_c_2,
            theta_v_ic,
            inv_ddqz_z_full,
            horz_idx,
            vert_idx,
            limited_area,
            igradp_method,
            horizontal_lower_01,
            horizontal_upper_01,
            horizontal_lower_02,
            horizontal_upper_02,
            horizontal_lower_03,
            horizontal_upper_03,
            n_lev,
            nflatlev,
            nflat_gradp,
        )
        if istep == 1
        else (
            z_rth_pr_1,
            z_rth_pr_2,
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
            z_dexner_dz_c_1,
        )
    )

    return z_exner_ex_pr, z_exner_ic


@field_operator
def _fused_mo_solve_nonhydro_stencils_01_to_13_restricted2(
    rho_nnow: Field[[CellDim, KDim], float],
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
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    istep: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    horizontal_lower_11: int32,
    horizontal_upper_11: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> Field[[CellDim, KDim], float]:
    # if istep == 1:

    (
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
        z_dexner_dz_c_1,
    ) = (
        _fused_mo_solve_nonhydro_stencils_01_to_13_predictor(
            rho_nnow,
            rho_ref_mc,
            theta_v_nnow,
            theta_ref_mc,
            z_rth_pr_1,
            z_rth_pr_2,
            z_theta_v_pr_ic,
            theta_ref_ic,
            d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc,
            wgtfacq_c_dsl,
            wgtfac_c,
            vwind_expl_wgt,
            exner_pr,
            d_exner_dz_ref_ic,
            ddqz_z_half,
            z_th_ddz_exner_c,
            rho_ic,
            z_exner_ic,
            exner_exfac,
            exner_nnow,
            exner_ref_mc,
            z_exner_ex_pr,
            z_dexner_dz_c_1,
            z_dexner_dz_c_2,
            theta_v_ic,
            inv_ddqz_z_full,
            horz_idx,
            vert_idx,
            limited_area,
            igradp_method,
            horizontal_lower_01,
            horizontal_upper_01,
            horizontal_lower_02,
            horizontal_upper_02,
            horizontal_lower_03,
            horizontal_upper_03,
            n_lev,
            nflatlev,
            nflat_gradp,
        )
        if istep == 1
        else (
            z_rth_pr_1,
            z_rth_pr_2,
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
            z_dexner_dz_c_1,
        )
    )

    return z_dexner_dz_c_1


@field_operator
def _fused_mo_solve_nonhydro_stencils_01_to_13(
    rho_nnow: Field[[CellDim, KDim], float],
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
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    istep: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    horizontal_lower_11: int32,
    horizontal_upper_11: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    # if istep == 1:
    (
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
        z_dexner_dz_c_1,
    ) = (
        _fused_mo_solve_nonhydro_stencils_01_to_13_predictor(
            rho_nnow,
            rho_ref_mc,
            theta_v_nnow,
            theta_ref_mc,
            z_rth_pr_1,
            z_rth_pr_2,
            z_theta_v_pr_ic,
            theta_ref_ic,
            d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc,
            wgtfacq_c_dsl,
            wgtfac_c,
            vwind_expl_wgt,
            exner_pr,
            d_exner_dz_ref_ic,
            ddqz_z_half,
            z_th_ddz_exner_c,
            rho_ic,
            z_exner_ic,
            exner_exfac,
            exner_nnow,
            exner_ref_mc,
            z_exner_ex_pr,
            z_dexner_dz_c_1,
            z_dexner_dz_c_2,
            theta_v_ic,
            inv_ddqz_z_full,
            horz_idx,
            vert_idx,
            limited_area,
            igradp_method,
            horizontal_lower_01,
            horizontal_upper_01,
            horizontal_lower_02,
            horizontal_upper_02,
            horizontal_lower_03,
            horizontal_upper_03,
            n_lev,
            nflatlev,
            nflat_gradp,
        )
        if istep == 1
        else (
            z_rth_pr_1,
            z_rth_pr_2,
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
            z_dexner_dz_c_1,
        )
    )
    # else:
    #
    #     (
    #         rho_ic,
    #         z_theta_v_pr_ic,
    #         theta_v_ic,
    #         z_th_ddz_exner_c,
    #     ) = _fused_mo_solve_nonhydro_stencils_01_to_13_corrector(
    #         w=w,
    #         w_concorr_c=w_concorr_c,
    #         ddqz_z_half=ddqz_z_half,
    #         rho_nnow=rho_nnow,
    #         rho_nvar=rho_nvar,
    #         theta_v_nnow=theta_v_nnow,
    #         theta_v_nvar=theta_v_nvar,
    #         wgtfac_c=wgtfac_c,
    #         theta_ref_mc=theta_ref_mc,
    #         vwind_expl_wgt=vwind_expl_wgt,
    #         exner_pr=exner_pr,
    #         d_exner_dz_ref_ic=d_exner_dz_ref_ic,
    #         rho_ic=rho_ic,
    #         z_theta_v_pr_ic=z_theta_v_pr_ic,
    #         theta_v_ic=theta_v_ic,
    #         z_th_ddz_exner_c=z_th_ddz_exner_c,
    #         dtime=dtime,
    #         wgt_nnow_rth=wgt_nnow_rth,
    #         wgt_nnew_rth=wgt_nnew_rth,
    #         horz_idx=horz_idx,
    #         vert_idx=vert_idx,
    #         horizontal_lower_11=horizontal_lower_11,
    #         horizontal_upper_11=horizontal_upper_11,
    #         n_lev=n_lev,
    #     )

    return (
        z_rth_pr_1,
        z_rth_pr_2,
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_solve_nonhydro_stencils_01_to_13(
    rho_nnow: Field[[CellDim, KDim], float],
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
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    istep: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    horizontal_lower_11: int32,
    horizontal_upper_11: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_mo_solve_nonhydro_stencils_01_to_13(
        rho_nnow,
        rho_ref_mc,
        theta_v_nnow,
        theta_ref_mc,
        z_rth_pr_1,
        z_rth_pr_2,
        z_theta_v_pr_ic,
        theta_ref_ic,
        d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc,
        wgtfacq_c_dsl,
        wgtfac_c,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_th_ddz_exner_c,
        rho_ic,
        z_exner_ic,
        exner_exfac,
        exner_nnow,
        exner_ref_mc,
        z_exner_ex_pr,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        theta_v_ic,
        inv_ddqz_z_full,
        horz_idx,
        vert_idx,
        limited_area,
        igradp_method,
        w,
        w_concorr_c,
        rho_nvar,
        theta_v_nvar,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        istep,
        horizontal_lower_01,
        horizontal_upper_01,
        horizontal_lower_02,
        horizontal_upper_02,
        horizontal_lower_03,
        horizontal_upper_03,
        horizontal_lower_11,
        horizontal_upper_11,
        n_lev,
        nflatlev,
        nflat_gradp,
        out=(
            z_rth_pr_1,
            z_rth_pr_2,
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )

    _fused_mo_solve_nonhydro_stencils_01_to_13_restricted(
        rho_nnow,
        rho_ref_mc,
        theta_v_nnow,
        theta_ref_mc,
        z_rth_pr_1,
        z_rth_pr_2,
        z_theta_v_pr_ic,
        theta_ref_ic,
        d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc,
        wgtfacq_c_dsl,
        wgtfac_c,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_th_ddz_exner_c,
        rho_ic,
        z_exner_ic,
        exner_exfac,
        exner_nnow,
        exner_ref_mc,
        z_exner_ex_pr,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        theta_v_ic,
        inv_ddqz_z_full,
        horz_idx,
        vert_idx,
        limited_area,
        igradp_method,
        w,
        w_concorr_c,
        rho_nvar,
        theta_v_nvar,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        istep,
        horizontal_lower_01,
        horizontal_upper_01,
        horizontal_lower_02,
        horizontal_upper_02,
        horizontal_lower_03,
        horizontal_upper_03,
        horizontal_lower_11,
        horizontal_upper_11,
        n_lev,
        nflatlev,
        nflat_gradp,
        out=(z_exner_ex_pr, z_exner_ic),
        # out=z_exner_ex_pr,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )

    _fused_mo_solve_nonhydro_stencils_01_to_13_restricted2(
        rho_nnow,
        rho_ref_mc,
        theta_v_nnow,
        theta_ref_mc,
        z_rth_pr_1,
        z_rth_pr_2,
        z_theta_v_pr_ic,
        theta_ref_ic,
        d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc,
        wgtfacq_c_dsl,
        wgtfac_c,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_th_ddz_exner_c,
        rho_ic,
        z_exner_ic,
        exner_exfac,
        exner_nnow,
        exner_ref_mc,
        z_exner_ex_pr,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        theta_v_ic,
        inv_ddqz_z_full,
        horz_idx,
        vert_idx,
        limited_area,
        igradp_method,
        w,
        w_concorr_c,
        rho_nvar,
        theta_v_nvar,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        istep,
        horizontal_lower_01,
        horizontal_upper_01,
        horizontal_lower_02,
        horizontal_upper_02,
        horizontal_lower_03,
        horizontal_upper_03,
        horizontal_lower_11,
        horizontal_upper_11,
        n_lev,
        nflatlev,
        nflat_gradp,
        out=z_dexner_dz_c_1,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )
