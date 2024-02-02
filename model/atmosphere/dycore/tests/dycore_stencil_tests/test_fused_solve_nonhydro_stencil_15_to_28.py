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

import numpy as np
import pytest
from gt4py.next import int32

from icon4py.model.atmosphere.dycore.fused_solve_nonhydro_stencil_15_to_28 import (
    fused_solve_nonhydro_stencil_15_to_28,
)
from icon4py.model.common.dimension import (
    C2E2CODim,
    CellDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
)
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    flatten_first_two_dims,
    random_field,
    random_mask,
    zero_field,
)

from model.atmosphere.dycore.tests.dycore_stencil_tests.test_mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_horizontal_advection_of_rho_and_theta import (
    compute_horizontal_advection_of_rho_and_theta_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_horizontal_gradient_of_extner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_extner_pressure_for_multiple_levels_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_graddiv2_of_vn import (
    compute_graddiv2_of_vn_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_apply_weighted_2nd_and_4th_order_divergence_damping import (
    apply_weighted_2nd_and_4th_order_divergence_damping_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn_numpy,
)


class TestFusedMoSolveNonHydroStencil15To28(StencilTest):
    PROGRAM = fused_solve_nonhydro_stencil_15_to_28
    OUTPUTS = ("z_rho_e", "z_theta_v_e", "z_gradh_exner", "vn", "z_graddiv_vn")

    # flake8: noqa: C901
    @classmethod
    def reference(
        cls,
        grid,
        geofac_grg_x,
        geofac_grg_y,
        p_vn,
        p_vt,
        pos_on_tplane_e_1,
        pos_on_tplane_e_2,
        primal_normal_cell_1,
        dual_normal_cell_1,
        primal_normal_cell_2,
        dual_normal_cell_2,
        rho_ref_me,
        theta_ref_me,
        z_rth_pr_1,
        z_rth_pr_2,
        ddxn_z_full,
        c_lin_e,
        z_exner_ex_pr,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        theta_v,
        ikoffset,
        zdiff_gradp,
        theta_v_ic,
        inv_ddqz_z_full,
        ipeidx_dsl,
        pg_exdist,
        hmask_dd3d,
        scalfac_dd3d,
        z_dwdz_dd,
        inv_dual_edge_length,
        ddt_vn_apc_ntl2,
        vn_nnow,
        ddt_vn_apc_ntl1,
        ddt_vn_phy,
        z_graddiv_vn,
        vn_incr,
        vn,
        z_rho_e,
        z_theta_v_e,
        z_gradh_exner,
        z_graddiv2_vn,
        z_hydro_corr,
        geofac_grdiv,
        scal_divdamp,
        bdy_divdamp,
        nudgecoeff_e,
        horz_idx,
        vert_idx,
        grav_o_cpd,
        p_dthalf,
        wgt_nnow_vel,
        wgt_nnew_vel,
        dtime,
        cpd,
        iau_wgt_dyn,
        is_iau_active,
        lhdiff_rcf,
        divdamp_fac,
        divdamp_fac_o2,
        divdamp_order,
        scal_divdamp_o2,
        limited_area,
        itime_scheme,
        istep,
        horizontal_lower_0,
        horizontal_upper_0,
        horizontal_lower_00,
        horizontal_upper_00,
        horizontal_lower_01,
        horizontal_upper_01,
        horizontal_lower_1,
        horizontal_upper_1,
        horizontal_lower_2,
        horizontal_upper_2,
        horizontal_lower_3,
        horizontal_upper_3,
        horizontal_lower_4,
        horizontal_upper_4,
        kstart_dd3d,
        nflatlev,
        nflat_gradp,
    ):
        iadv_rhotheta = 2
        horz_idx = horz_idx[:, np.newaxis]
        horizontal_lower = horizontal_lower_0
        horizontal_upper = horizontal_upper_0
        nlev = 10
        z_grad_rth_1 = zero_field(grid, CellDim, KDim)
        z_grad_rth_2 = zero_field(grid, CellDim, KDim)
        z_grad_rth_3 = zero_field(grid, CellDim, KDim)
        z_grad_rth_4 = zero_field(grid, CellDim, KDim)

        if istep == 1:
            if iadv_rhotheta == 2:
                # Compute Green-Gauss gradients for rho and theta
                (
                    z_grad_rth_1,
                    z_grad_rth_2,
                    z_grad_rth_3,
                    z_grad_rth_4,
                ) = mo_math_gradients_grad_green_gauss_cell_dsl_numpy(
                    grid=grid,
                    p_ccpr1=z_rth_pr_1,
                    p_ccpr2=z_rth_pr_2,
                    geofac_grg_x=geofac_grg_x,
                    geofac_grg_y=geofac_grg_y,
                )

            if iadv_rhotheta <= 2:
                (tmp_0_0, tmp_0_1) = (horizontal_lower_00, horizontal_upper_00)
                #if idiv_method == 1:
                (tmp_0_0, tmp_0_1) = (horizontal_lower_01, horizontal_upper_01)

                z_rho_e = np.where(
                    (tmp_0_0 <= horz_idx) & (horz_idx < tmp_0_1),
                    np.zeros_like(z_rho_e),
                    z_rho_e,
                )

                z_theta_v_e = np.where(
                    (tmp_0_0 <= horz_idx) & (horz_idx < tmp_0_1),
                    np.zeros_like(z_rho_e),
                    z_theta_v_e,
                )

                # initialize also nest boundary points with zero
                if limited_area:
                    z_rho_e = np.where(
                        (horizontal_lower_4 <= horz_idx) & (horz_idx < horizontal_upper_4),
                        np.zeros_like(z_rho_e),
                        z_rho_e,
                    )

                    z_theta_v_e = np.where(
                        (horizontal_lower_4 <= horz_idx) & (horz_idx < horizontal_upper_4),
                        np.zeros_like(z_rho_e),
                        z_theta_v_e,
                    )

                if iadv_rhotheta == 2:
                    # Compute upwind-biased values for rho and theta starting from centered differences
                    # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                    # at a second-order accurate FV discretization, but twice the length is needed for numerical stability
                    (z_rho_e, z_theta_v_e) = np.where(
                        (horizontal_lower_1 <= horz_idx) & (horz_idx < horizontal_upper_1),
                        compute_horizontal_advection_of_rho_and_theta_numpy(
                            grid=grid,
                            p_vn=p_vn,
                            p_vt=p_vt,
                            pos_on_tplane_e_1=pos_on_tplane_e_1,
                            pos_on_tplane_e_2=pos_on_tplane_e_2,
                            primal_normal_cell_1=primal_normal_cell_1,
                            dual_normal_cell_1=dual_normal_cell_1,
                            primal_normal_cell_2=primal_normal_cell_2,
                            dual_normal_cell_2=dual_normal_cell_2,
                            p_dthalf=p_dthalf,
                            rho_ref_me=rho_ref_me,
                            theta_ref_me=theta_ref_me,
                            z_grad_rth_1=z_grad_rth_1,
                            z_grad_rth_2=z_grad_rth_2,
                            z_grad_rth_3=z_grad_rth_3,
                            z_grad_rth_4=z_grad_rth_4,
                            z_rth_pr_1=z_rth_pr_1,
                            z_rth_pr_2=z_rth_pr_2,
                        ),
                        (z_rho_e, z_theta_v_e),
                    )

            # Remaining computations at edge points
            z_gradh_exner = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx < nflatlev),
                compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_numpy(
                    grid=grid,
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=z_exner_ex_pr,
                ),
                z_gradh_exner,
            )

            #if igradp_method == 3:
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
            z_gradh_exner = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx >= nflatlev)
                & (vert_idx < (nflat_gradp + int32(1))),
                compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_numpy(
                    grid=grid,
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=z_exner_ex_pr,
                    ddxn_z_full=ddxn_z_full,
                    c_lin_e=c_lin_e,
                    z_dexner_dz_c_1=z_dexner_dz_c_1,
                ),
                z_gradh_exner,
            )

            new_shape = list(grid.e2c.shape)
            new_shape.append(nlev)
            new_shape = tuple(new_shape)
            z_gradh_exner = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx >= (nflat_gradp + int32(1))),
                compute_horizontal_gradient_of_extner_pressure_for_multiple_levels_numpy(
                    grid=grid,
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=z_exner_ex_pr,
                    zdiff_gradp=zdiff_gradp.reshape(new_shape),
                    ikoffset=ikoffset.reshape(new_shape),
                    z_dexner_dz_c_1=z_dexner_dz_c_1,
                    z_dexner_dz_c_2=z_dexner_dz_c_2,
                ),
                z_gradh_exner,
            )
        # compute hydrostatically approximated correction term that replaces downward extrapolation
            #if igradp_method == 3:
            z_hydro_corr = compute_hydrostatic_correction_term_numpy(
                grid=grid,
                theta_v=theta_v,
                ikoffset=ikoffset.reshape(new_shape),
                zdiff_gradp=zdiff_gradp.reshape(new_shape),
                theta_v_ic=theta_v_ic,
                inv_ddqz_z_full=inv_ddqz_z_full,
                inv_dual_edge_length=inv_dual_edge_length,
                grav_o_cpd=grav_o_cpd,
            )

            #if igradp_method == 3:
            z_gradh_exner = np.where(
                (horizontal_lower_3 <= horz_idx) & (horz_idx < horizontal_upper_3),
                apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_numpy(
                    grid=grid,
                    ipeidx_dsl=ipeidx_dsl,
                    pg_exdist=pg_exdist,
                    z_hydro_corr=z_hydro_corr,
                    z_gradh_exner=z_gradh_exner,
                ),
                z_gradh_exner,
            )

            vn = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                add_temporal_tendencies_to_vn_numpy(
                    grid=grid,
                    vn_nnow=vn_nnow,
                    ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
                    ddt_vn_phy=ddt_vn_phy,
                    z_theta_v_e=z_theta_v_e,
                    z_gradh_exner=z_gradh_exner,
                    dtime=dtime,
                    cpd=cpd,
                ),
                vn,
            )

            if is_iau_active:
                vn = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    add_analysis_increments_to_vn_numpy(
                        grid=grid,
                        vn_incr=vn_incr,
                        vn=vn,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    vn,
                )

        else:

            z_graddiv_vn = np.where(
                (horizontal_lower_2 <= horz_idx)
                & (horz_idx < horizontal_upper_2)
                & (vert_idx >= kstart_dd3d),
                add_vertical_wind_derivative_to_divergence_damping_numpy(
                    grid=grid,
                    hmask_dd3d=hmask_dd3d,
                    scalfac_dd3d=scalfac_dd3d,
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_dwdz_dd=z_dwdz_dd,
                    z_graddiv_vn=z_graddiv_vn,
                ),
                z_graddiv_vn,
            )

            if itime_scheme == 4:
                vn = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy(
                        grid=grid,
                        vn_nnow=vn,
                        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
                        ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
                        ddt_vn_phy=ddt_vn_phy,
                        z_theta_v_e=z_theta_v_e,
                        z_gradh_exner=z_gradh_exner,
                        dtime=dtime,
                        wgt_nnow_vel=wgt_nnow_vel,
                        wgt_nnew_vel=wgt_nnew_vel,
                        cpd=cpd,
                    ),
                    vn,
                )

            if lhdiff_rcf and (divdamp_order == 24 or divdamp_order == 4):
                # verified for e-10
                z_graddiv2_vn = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    compute_graddiv2_of_vn_numpy(
                        grid=grid,
                        geofac_grdiv=geofac_grdiv,
                        z_graddiv_vn=z_graddiv_vn,
                    ),
                    z_graddiv2_vn,
                )

            if lhdiff_rcf:
                if divdamp_order == 24 and scal_divdamp_o2 > 1.0e-6:
                    vn = np.where(
                        (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                        apply_2nd_order_divergence_damping_numpy(
                            grid=grid,
                            z_graddiv_vn=z_graddiv_vn,
                            vn=vn,
                            scal_divdamp_o2=scal_divdamp_o2,
                        ),
                        vn,
                    )

                if divdamp_order == 24 and divdamp_fac_o2 <= 4 * divdamp_fac:
                    if limited_area:
                        vn = np.where(
                            (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                            apply_weighted_2nd_and_4th_order_divergence_damping_numpy(
                                grid=grid,
                                scal_divdamp=scal_divdamp,
                                bdy_divdamp=bdy_divdamp,
                                nudgecoeff_e=nudgecoeff_e,
                                z_graddiv2_vn=z_graddiv2_vn,
                                vn=vn,
                            ),
                            vn,
                        )
                    else:
                        vn = np.where(
                            (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                            mo_solve_nonhydro_4th_order_divdamp_numpy(
                                grid=grid,
                                scal_divdamp=scal_divdamp,
                                z_graddiv2_vn=z_graddiv2_vn,
                                vn=vn,
                            ),
                            vn,
                        )

            if is_iau_active:
                vn = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    add_analysis_increments_to_vn_numpy(
                        grid=grid,
                        vn_incr=vn_incr,
                        vn=vn,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    vn,
                )

        return dict(
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn=vn,
            z_graddiv_vn=z_graddiv_vn,
        )

    @pytest.fixture
    def input_data(self, grid):
        geofac_grg_x = random_field(grid, CellDim, C2E2CODim)
        geofac_grg_y = random_field(grid, CellDim, C2E2CODim)
        p_vn = random_field(grid, EdgeDim, KDim)
        p_vt = random_field(grid, EdgeDim, KDim)
        pos_on_tplane_e_1 = random_field(grid, ECDim)
        pos_on_tplane_e_2 = random_field(grid, ECDim)
        primal_normal_cell_1 = random_field(grid, ECDim)
        dual_normal_cell_1 = random_field(grid, ECDim)
        primal_normal_cell_2 = random_field(grid, ECDim)
        dual_normal_cell_2 = random_field(grid, ECDim)
        rho_ref_me = random_field(grid, EdgeDim, KDim)
        theta_ref_me = random_field(grid, EdgeDim, KDim)
        z_rth_pr_1 = random_field(grid, CellDim, KDim)
        z_rth_pr_2 = random_field(grid, CellDim, KDim)
        ddxn_z_full = random_field(grid, EdgeDim, KDim)
        c_lin_e = random_field(grid, EdgeDim, E2CDim)
        z_exner_ex_pr = random_field(grid, CellDim, KDim)
        z_dexner_dz_c_1 = random_field(grid, CellDim, KDim)
        z_dexner_dz_c_2 = random_field(grid, CellDim, KDim)
        z_hydro_corr = random_field(grid, EdgeDim, KDim)
        theta_v = random_field(grid, CellDim, KDim)
        zdiff_gradp = random_field(grid, ECDim, KDim)
        theta_v_ic = random_field(grid, CellDim, KDim)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim)
        ipeidx_dsl = random_mask(grid, EdgeDim, KDim)
        pg_exdist = random_field(grid, EdgeDim, KDim)
        hmask_dd3d = random_field(grid, EdgeDim)
        scalfac_dd3d = random_field(grid, KDim)
        z_dwdz_dd = random_field(grid, CellDim, KDim)
        inv_dual_edge_length = random_field(grid, EdgeDim)
        ddt_vn_apc_ntl2 = random_field(grid, EdgeDim, KDim)
        vn_nnow = random_field(grid, EdgeDim, KDim)
        ddt_vn_apc_ntl1 = random_field(grid, EdgeDim, KDim)
        ddt_vn_phy = random_field(grid, EdgeDim, KDim)
        z_graddiv_vn = random_field(grid, EdgeDim, KDim)
        vn_incr = random_field(grid, EdgeDim, KDim)
        vn = random_field(grid, EdgeDim, KDim)
        z_theta_v_e = random_field(grid, EdgeDim, KDim)
        z_gradh_exner = random_field(grid, EdgeDim, KDim)
        z_graddiv2_vn = random_field(grid, EdgeDim, KDim)
        z_rho_e = random_field(grid, EdgeDim, KDim)
        geofac_grdiv = random_field(grid, EdgeDim, E2C2EODim)
        scal_divdamp = random_field(grid, KDim)
        bdy_divdamp = random_field(grid, KDim)
        nudgecoeff_e = random_field(grid, EdgeDim)

        ikoffset = zero_field(grid, EdgeDim, E2CDim, KDim, dtype=int32)
        rng = np.random.default_rng()
        k_levels = 10

        for k in range(k_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=k_levels - k - 1,
                size=(ikoffset.shape[0], ikoffset.shape[1]),
            )
        ikoffset = flatten_first_two_dims(ECDim, KDim, field=ikoffset)

        vert_idx = zero_field(grid, KDim, dtype=int32)
        for level in range(k_levels):
            vert_idx[level] = level

        horz_idx = zero_field(grid, EdgeDim, dtype=int32)
        for edge in range(grid.num_edges):
            horz_idx[edge] = edge

        grav_o_cpd = 9.80665 / 1004.64
        dtime = 0.9
        p_dthalf = 0.5 * dtime
        wgt_nnow_vel = 0.25
        wgt_nnew_vel = 0.75
        cpd = 1004.64
        iau_wgt_dyn = 1.0
        is_iau_active = False
        lhdiff_rcf = True
        divdamp_fac = 0.004
        divdamp_fac_o2 = 0.032
        divdamp_order = 24
        scal_divdamp_o2 = 194588.14247428576
        limited_area = True
        itime_scheme = 4
        istep = 2
        horizontal_lower_0 = 5387
        horizontal_upper_0 = 31558
        horizontal_lower_00 = 31558
        horizontal_upper_00 = 31558
        horizontal_lower_01 = 31558
        horizontal_upper_01 = 31558
        horizontal_lower_1 = 3777
        horizontal_upper_1 = 31558
        horizontal_lower_2 = 3777
        horizontal_upper_2 = 31558
        horizontal_lower_3 = 5387
        horizontal_upper_3 = 31558
        horizontal_lower_4 = 0
        horizontal_upper_4 = 31558
        kstart_dd3d = 0
        nlev = k_levels
        nflatlev = 4
        nflat_gradp = 27

        return dict(
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            z_exner_ex_pr=z_exner_ex_pr,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
            theta_v=theta_v,
            ikoffset=ikoffset,
            zdiff_gradp=zdiff_gradp,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            z_dwdz_dd=z_dwdz_dd,
            inv_dual_edge_length=inv_dual_edge_length,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            vn_incr=vn_incr,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            z_hydro_corr=z_hydro_corr,
            z_graddiv_vn=z_graddiv_vn,
            vn=vn,
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            z_graddiv2_vn=z_graddiv2_vn,
            geofac_grdiv=geofac_grdiv,
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            grav_o_cpd=grav_o_cpd,
            p_dthalf=p_dthalf,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            dtime=dtime,
            cpd=cpd,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            lhdiff_rcf=lhdiff_rcf,
            divdamp_fac=divdamp_fac,
            divdamp_fac_o2=divdamp_fac_o2,
            divdamp_order=divdamp_order,
            scal_divdamp_o2=scal_divdamp_o2,
            limited_area=limited_area,
            itime_scheme=itime_scheme,
            istep=istep,
            horizontal_lower_0=horizontal_lower_0,
            horizontal_upper_0=horizontal_upper_0,
            horizontal_lower_00=horizontal_lower_00,
            horizontal_upper_00=horizontal_upper_00,
            horizontal_lower_01=horizontal_lower_01,
            horizontal_upper_01=horizontal_upper_01,
            horizontal_lower_1=horizontal_lower_1,
            horizontal_upper_1=horizontal_upper_1,
            horizontal_lower_2=horizontal_lower_2,
            horizontal_upper_2=horizontal_upper_2,
            horizontal_lower_3=horizontal_lower_3,
            horizontal_upper_3=horizontal_upper_3,
            horizontal_lower_4=horizontal_lower_4,
            horizontal_upper_4=horizontal_upper_4,
            kstart_dd3d=kstart_dd3d,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
