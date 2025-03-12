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
import numpy as np
import pytest
from .test_add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn_numpy,
)
from .test_add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn_numpy,
)
from .test_add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy,
)
from .test_add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping_numpy,
)
from .test_apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping_numpy,
)
from .test_apply_4th_order_divergence_damping import (
    apply_4th_order_divergence_damping_numpy,
)
from .test_apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_numpy,
)
from .test_apply_weighted_2nd_and_4th_order_divergence_damping import (
    apply_weighted_2nd_and_4th_order_divergence_damping_numpy,
)
from .test_compute_graddiv2_of_vn import (
    compute_graddiv2_of_vn_numpy,
)
from .test_compute_horizontal_advection_of_rho_and_theta import (
    compute_horizontal_advection_of_rho_and_theta_numpy,
)
from .test_compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_numpy,
)
from .test_compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels_numpy,
)
from .test_compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_numpy,
)
from .test_compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term_numpy,
)
from .test_mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl_numpy,
)

import icon4py.model.common.type_alias as ta
from icon4py.model.atmosphere.dycore.fused_solve_nonhydro_stencil_15_to_28 import (
    fused_solve_nonhydro_stencil_15_to_28,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestFusedMoSolveNonHydroStencil15To28(StencilTest):
    PROGRAM = fused_solve_nonhydro_stencil_15_to_28
    OUTPUTS = ("z_rho_e", "z_theta_v_e", "z_gradh_exner", "vn", "z_graddiv_vn")

    # flake8: noqa: C901
    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        p_vn: np.ndarray,
        p_vt: np.ndarray,
        pos_on_tplane_e_1: np.ndarray,
        pos_on_tplane_e_2: np.ndarray,
        primal_normal_cell_1: np.ndarray,
        dual_normal_cell_1: np.ndarray,
        primal_normal_cell_2: np.ndarray,
        dual_normal_cell_2: np.ndarray,
        rho_ref_me: np.ndarray,
        theta_ref_me: np.ndarray,
        z_rth_pr_1: np.ndarray,
        z_rth_pr_2: np.ndarray,
        ddxn_z_full: np.ndarray,
        c_lin_e: np.ndarray,
        z_exner_ex_pr: np.ndarray,
        z_dexner_dz_c_1: np.ndarray,
        z_dexner_dz_c_2: np.ndarray,
        theta_v: np.ndarray,
        ikoffset: np.ndarray,
        zdiff_gradp: np.ndarray,
        theta_v_ic: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        ipeidx_dsl: np.ndarray,
        pg_exdist: np.ndarray,
        hmask_dd3d: np.ndarray,
        scalfac_dd3d: np.ndarray,
        z_dwdz_dd: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        ddt_vn_apc_ntl2: np.ndarray,
        ddt_vn_apc_ntl1: np.ndarray,
        ddt_vn_phy: np.ndarray,
        vn_incr: np.ndarray,
        horz_idx: np.ndarray,
        vert_idx: np.ndarray,
        bdy_divdamp: np.ndarray,
        nudgecoeff_e: np.ndarray,
        z_hydro_corr: np.ndarray,
        geofac_grdiv: np.ndarray,
        z_graddiv2_vn: np.ndarray,
        scal_divdamp: np.ndarray,
        z_rho_e: np.ndarray,
        z_theta_v_e: np.ndarray,
        z_gradh_exner: np.ndarray,
        vn: np.ndarray,
        z_graddiv_vn: np.ndarray,
        divdamp_fac: ta.wpfloat,
        divdamp_fac_o2: ta.wpfloat,
        wgt_nnow_vel: ta.wpfloat,
        wgt_nnew_vel: ta.wpfloat,
        dtime: ta.wpfloat,
        cpd: ta.wpfloat,
        iau_wgt_dyn: ta.wpfloat,
        is_iau_active: gtx.int32,
        itime_scheme: gtx.int32,
        p_dthalf: gtx.int32,
        grav_o_cpd: gtx.int32,
        limited_area: gtx.int32,
        divdamp_order: gtx.int32,
        scal_divdamp_o2: ta.wpfloat,
        istep: gtx.int32,
        start_edge_halo_level_2: gtx.int32,
        end_edge_halo_level_2: gtx.int32,
        start_edge_lateral_boundary: gtx.int32,
        end_edge_halo: gtx.int32,
        start_edge_lateral_boundary_level_7: gtx.int32,
        start_edge_nudging_level_2: gtx.int32,
        end_edge_local: gtx.int32,
        end_edge_end: gtx.int32,
        iadv_rhotheta: gtx.int32,
        igradp_method: gtx.int32,
        MIURA: gtx.int32,
        TAYLOR_HYDRO: gtx.int32,
        nlev: gtx.int32,
        kstart_dd3d: gtx.int32,
        COMBINED: gtx.int32,
        FOURTH_ORDER: gtx.int32,
        nflatlev: gtx.int32,
        nflat_gradp: gtx.int32,
    ) -> dict:
        horz_idx = horz_idx[:, np.newaxis]

        z_grad_rth_1 = np.zeros(theta_v.shape)
        z_grad_rth_2 = np.zeros(theta_v.shape)
        z_grad_rth_3 = np.zeros(theta_v.shape)
        z_grad_rth_4 = np.zeros(theta_v.shape)

        if istep == 1:
            if iadv_rhotheta == MIURA:
                # Compute Green-Gauss gradients for rho and theta
                (
                    z_grad_rth_1,
                    z_grad_rth_2,
                    z_grad_rth_3,
                    z_grad_rth_4,
                ) = mo_math_gradients_grad_green_gauss_cell_dsl_numpy(
                    connectivities=connectivities,
                    p_ccpr1=z_rth_pr_1,
                    p_ccpr2=z_rth_pr_2,
                    geofac_grg_x=geofac_grg_x,
                    geofac_grg_y=geofac_grg_y,
                )

            if iadv_rhotheta <= 2:
                # if idiv_method == 1:
                (z_rho_e, z_theta_v_e) = np.where(
                    (horz_idx >= start_edge_halo_level_2) & (horz_idx < end_edge_halo_level_2),
                    (0.0, 0.0),
                    (z_rho_e, z_theta_v_e),
                )

                # initialize also nest boundary points with zero
                if limited_area:
                    (z_rho_e, z_theta_v_e) = np.where(
                        (horz_idx >= start_edge_lateral_boundary) & (horz_idx < end_edge_halo),
                        (0.0, 0.0),
                        (z_rho_e, z_theta_v_e),
                    )

                if iadv_rhotheta == MIURA:
                    # Compute upwind-biased values for rho and theta starting from centered differences
                    # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                    # at a second-order accurate FV discretization, but twice the length is needed for numerical stability
                    (z_rho_e, z_theta_v_e) = np.where(
                        (start_edge_lateral_boundary_level_7 <= horz_idx)
                        & (horz_idx < end_edge_halo),
                        compute_horizontal_advection_of_rho_and_theta_numpy(
                            connectivities=connectivities,
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
                (start_edge_nudging_level_2 <= horz_idx)
                & (horz_idx < end_edge_local)
                & (vert_idx < nflatlev),
                compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_numpy(
                    connectivities=connectivities,
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=z_exner_ex_pr,
                ),
                z_gradh_exner,
            )

            if igradp_method == TAYLOR_HYDRO:
                # horizontal gradient of Exner pressure, including metric correction
                # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
                z_gradh_exner = np.where(
                    (start_edge_nudging_level_2 <= horz_idx)
                    & (horz_idx < end_edge_local)
                    & (vert_idx >= nflatlev)
                    & (vert_idx < (nflat_gradp + gtx.int32(1))),
                    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_numpy(
                        connectivities=connectivities,
                        inv_dual_edge_length=inv_dual_edge_length,
                        z_exner_ex_pr=z_exner_ex_pr,
                        ddxn_z_full=ddxn_z_full,
                        c_lin_e=c_lin_e,
                        z_dexner_dz_c_1=z_dexner_dz_c_1,
                    ),
                    z_gradh_exner,
                )

                new_shape = list(connectivities[dims.E2CDim].shape)
                new_shape.append(nlev)
                new_shape = tuple(new_shape)
                z_gradh_exner = np.where(
                    (start_edge_nudging_level_2 <= horz_idx)
                    & (horz_idx < end_edge_local)
                    & (vert_idx >= (nflat_gradp + gtx.int32(1))),
                    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels_numpy(
                        connectivities=connectivities,
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
                # if igradp_method == 3:
                z_hydro_corr = np.where(
                    (start_edge_nudging_level_2 <= horz_idx)
                    & (horz_idx < end_edge_local)
                    & (vert_idx >= (nlev - 1)),
                    compute_hydrostatic_correction_term_numpy(
                        connectivities=connectivities,
                        theta_v=theta_v,
                        ikoffset=ikoffset.reshape(new_shape),
                        zdiff_gradp=zdiff_gradp.reshape(new_shape),
                        theta_v_ic=theta_v_ic,
                        inv_ddqz_z_full=inv_ddqz_z_full,
                        inv_dual_edge_length=inv_dual_edge_length,
                        grav_o_cpd=grav_o_cpd,
                    ),
                    z_hydro_corr,
                )

                hydro_corr_horizontal = z_hydro_corr[:, nlev - 1]
                # if igradp_method == 3:
                z_gradh_exner = np.where(
                    (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_end),
                    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_numpy(
                        ipeidx_dsl=ipeidx_dsl,
                        pg_exdist=pg_exdist,
                        z_hydro_corr=hydro_corr_horizontal,
                        z_gradh_exner=z_gradh_exner,
                    ),
                    z_gradh_exner,
                )

            vn = np.where(
                (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                add_temporal_tendencies_to_vn_numpy(
                    vn_nnow=vn,
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
                    (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                    add_analysis_increments_to_vn_numpy(
                        vn_incr=vn_incr,
                        vn=vn,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    vn,
                )

        else:
            z_graddiv_vn = np.where(
                (start_edge_lateral_boundary_level_7 <= horz_idx)
                & (horz_idx < end_edge_halo_level_2)
                & (vert_idx >= kstart_dd3d),
                add_vertical_wind_derivative_to_divergence_damping_numpy(
                    connectivities=connectivities,
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
                    (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy(
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

            if divdamp_order == COMBINED or divdamp_order == FOURTH_ORDER:
                # verified for e-10
                z_graddiv2_vn = np.where(
                    (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                    compute_graddiv2_of_vn_numpy(
                        connectivities=connectivities,
                        geofac_grdiv=geofac_grdiv,
                        z_graddiv_vn=z_graddiv_vn,
                    ),
                    z_graddiv2_vn,
                )

            if True:
                if divdamp_order == COMBINED and scal_divdamp_o2 > 1.0e-6:
                    vn = np.where(
                        (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                        apply_2nd_order_divergence_damping_numpy(
                            z_graddiv_vn=z_graddiv_vn,
                            vn=vn,
                            scal_divdamp_o2=scal_divdamp_o2,
                        ),
                        vn,
                    )

                if divdamp_order == COMBINED and divdamp_fac_o2 <= 4 * divdamp_fac:
                    if limited_area:
                        vn = np.where(
                            (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                            apply_weighted_2nd_and_4th_order_divergence_damping_numpy(
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
                            (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                            apply_4th_order_divergence_damping_numpy(
                                scal_divdamp=scal_divdamp,
                                z_graddiv2_vn=z_graddiv2_vn,
                                vn=vn,
                            ),
                            vn,
                        )

            if is_iau_active:
                vn = np.where(
                    (start_edge_nudging_level_2 <= horz_idx) & (horz_idx < end_edge_local),
                    add_analysis_increments_to_vn_numpy(
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
        geofac_grg_x = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)
        geofac_grg_y = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        pos_on_tplane_e_1 = data_alloc.random_field(grid, dims.ECDim)
        pos_on_tplane_e_2 = data_alloc.random_field(grid, dims.ECDim)
        primal_normal_cell_x = data_alloc.random_field(grid, dims.ECDim)
        dual_normal_cell_x = data_alloc.random_field(grid, dims.ECDim)
        primal_normal_cell_y = data_alloc.random_field(grid, dims.ECDim)
        dual_normal_cell_y = data_alloc.random_field(grid, dims.ECDim)
        rho_ref_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_ref_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_rth_pr_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_rth_pr_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        z_exner_ex_pr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_dexner_dz_c_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_dexner_dz_c_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_hydro_corr = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        zdiff_gradp = data_alloc.random_field(grid, dims.ECDim, dims.KDim)
        theta_v_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ipeidx_dsl = data_alloc.random_mask(grid, dims.EdgeDim, dims.KDim)
        pg_exdist = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        hmask_dd3d = data_alloc.random_field(grid, dims.EdgeDim)
        scalfac_dd3d = data_alloc.random_field(grid, dims.KDim)
        z_dwdz_dd = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        ddt_vn_apc_ntl2 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddt_vn_apc_ntl1 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddt_vn_phy = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_graddiv_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        vn_incr = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_gradh_exner = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_graddiv2_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_rho_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        scal_divdamp = data_alloc.random_field(grid, dims.KDim)
        bdy_divdamp = data_alloc.random_field(grid, dims.KDim)
        nudgecoeff_e = data_alloc.random_field(grid, dims.EdgeDim)

        ikoffset = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
        )
        rng = np.random.default_rng()
        k_levels = 10

        for k in range(k_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=k_levels - k - 1,
                size=(ikoffset.asnumpy().shape[0], ikoffset.asnumpy().shape[1]),
            )
        ikoffset = data_alloc.flatten_first_two_dims(dims.ECDim, dims.KDim, field=ikoffset)

        vert_idx = data_alloc.index_field(dim=dims.KDim, grid=grid)
        horz_idx = data_alloc.index_field(dim=dims.EdgeDim, grid=grid)

        grav_o_cpd = 9.80665 / 1004.64
        dtime = 0.9
        wgt_nnew_vel = 0.75
        wgt_nnow_vel = 0.25
        cpd = 1004.64
        iau_wgt_dyn = 1.0
        is_iau_active = False
        divdamp_fac = 0.004
        divdamp_fac_o2 = 0.032
        divdamp_order = 24
        scal_divdamp_o2 = 194588.14247428576
        limited_area = True
        itime_scheme = 4
        istep = 2
        iadv_rhotheta = 2
        igradp_method = 3
        MIURA = 2
        TAYLOR_HYDRO = 3
        COMBINED = 24
        FOURTH_ORDER = 4
        edge_domain = h_grid.domain(dims.EdgeDim)

        start_edge_halo_level_2 = grid.start_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        end_edge_halo_level_2 = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        start_edge_lateral_boundary = grid.end_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY))
        end_edge_halo = grid.end_index(edge_domain(h_grid.Zone.HALO))
        start_edge_lateral_boundary_level_7 = grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        start_edge_nudging_level_2 = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        end_edge_local = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        end_edge_end = grid.num_edges  # TODO: check
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
            primal_normal_cell_1=primal_normal_cell_x,
            dual_normal_cell_1=dual_normal_cell_x,
            primal_normal_cell_2=primal_normal_cell_y,
            dual_normal_cell_2=dual_normal_cell_y,
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
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            vn_incr=vn_incr,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            z_hydro_corr=z_hydro_corr,
            geofac_grdiv=geofac_grdiv,
            z_graddiv2_vn=z_graddiv2_vn,
            scal_divdamp=scal_divdamp,
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn=vn,
            z_graddiv_vn=z_graddiv_vn,
            divdamp_fac=divdamp_fac,
            divdamp_fac_o2=divdamp_fac_o2,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            dtime=dtime,
            cpd=cpd,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            itime_scheme=itime_scheme,
            p_dthalf=(0.5 * dtime),
            grav_o_cpd=grav_o_cpd,
            limited_area=limited_area,
            divdamp_order=divdamp_order,
            scal_divdamp_o2=scal_divdamp_o2,
            istep=istep,
            start_edge_halo_level_2=start_edge_halo_level_2,
            end_edge_halo_level_2=end_edge_halo_level_2,
            start_edge_lateral_boundary=start_edge_lateral_boundary,
            end_edge_halo=end_edge_halo,
            start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
            start_edge_nudging_level_2=start_edge_nudging_level_2,
            end_edge_local=end_edge_local,
            end_edge_end=end_edge_end,
            iadv_rhotheta=iadv_rhotheta,
            igradp_method=igradp_method,
            MIURA=MIURA,
            TAYLOR_HYDRO=TAYLOR_HYDRO,
            nlev=nlev,
            kstart_dd3d=kstart_dd3d,
            COMBINED=COMBINED,
            FOURTH_ORDER=FOURTH_ORDER,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
