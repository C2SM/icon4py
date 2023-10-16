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
from gt4py.next import np_as_located_field
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn

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

from test_mo_math_gradients_grad_green_gauss_cell_dsl import \
    mo_math_gradients_grad_green_gauss_cell_dsl_numpy
from test_mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp_numpy,
)
from test_mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1 import (
    mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1_numpy,
)
from test_mo_solve_nonhydro_stencil_17 import mo_solve_nonhydro_stencil_17_numpy
from test_mo_solve_nonhydro_stencil_18 import mo_solve_nonhydro_stencil_18_numpy
from test_mo_solve_nonhydro_stencil_19 import mo_solve_nonhydro_stencil_19_numpy
from test_mo_solve_nonhydro_stencil_20 import mo_solve_nonhydro_stencil_20_numpy
from test_mo_solve_nonhydro_stencil_21 import mo_solve_nonhydro_stencil_21_numpy
from test_mo_solve_nonhydro_stencil_22 import mo_solve_nonhydro_stencil_22_numpy
from test_mo_solve_nonhydro_stencil_23 import mo_solve_nonhydro_stencil_23_numpy
from test_mo_solve_nonhydro_stencil_24 import mo_solve_nonhydro_stencil_24_numpy
from test_mo_solve_nonhydro_stencil_25 import mo_solve_nonhydro_stencil_25_numpy
from test_mo_solve_nonhydro_stencil_26 import mo_solve_nonhydro_stencil_26_numpy
from test_mo_solve_nonhydro_stencil_27 import mo_solve_nonhydro_stencil_27_numpy
from test_mo_solve_nonhydro_stencil_28 import mo_solve_nonhydro_stencil_28_numpy


class TestFusedMoSolveNonHydroStencil15To28(StencilTest):
    PROGRAM = fused_solve_nonhydro_stencil_15_to_28.with_backend(run_gtfn)
    OUTPUTS = ("z_rho_e", "z_theta_v_e", "z_gradh_exner", "vn", "z_graddiv_vn")

    @classmethod
    def reference(
        cls,
        mesh,
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
        idiv_method,
        igradp_method,
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
        horizontal_lower,
        horizontal_upper,
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
        nlev,
        nflatlev,
        nflat_gradp,
    ):
        iadv_rhotheta = 2
        horz_idx = horz_idx[:, np.newaxis]
        z_grad_rth_1 = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_grad_rth_2 = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_grad_rth_3 = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_grad_rth_4 = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))

        if istep == 1:
            if iadv_rhotheta == 2:
                # Compute Green-Gauss gradients for rho and theta
                (
                    z_grad_rth_1,
                    z_grad_rth_2,
                    z_grad_rth_3,
                    z_grad_rth_4,
                ) = mo_math_gradients_grad_green_gauss_cell_dsl_numpy(
                    mesh=mesh,
                    p_ccpr1=z_rth_pr_1,
                    p_ccpr2=z_rth_pr_2,
                    geofac_grg_x=geofac_grg_x,
                    geofac_grg_y=geofac_grg_y,
                )

            if iadv_rhotheta <= 2:
                (tmp_0_0, tmp_0_1) = (horizontal_lower_00, horizontal_upper_00)
                if idiv_method == 1:
                    (tmp_0_0, tmp_0_1) = (horizontal_lower_01, horizontal_upper_01)

                z_rho_e = np.where(
                    (tmp_0_0 < horz_idx)
                    & (horz_idx < tmp_0_1)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    np.zeros_like(z_rho_e),
                    z_rho_e,
                )

                z_theta_v_e = np.where(
                    (tmp_0_0 < horz_idx)
                    & (horz_idx < tmp_0_1)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    np.zeros_like(z_rho_e),
                    z_theta_v_e,
                )

                # initialize also nest boundary points with zero
                if limited_area:
                    z_rho_e = np.where(
                        (horizontal_lower_4 < horz_idx)
                        & (horz_idx < horizontal_upper_4)
                        & (vert_idx > int32(0))
                        & (vert_idx < nlev),
                        np.zeros_like(z_rho_e),
                        z_rho_e,
                    )

                    z_theta_v_e = np.where(
                        (horizontal_lower_4 < horz_idx)
                        & (horz_idx < horizontal_upper_4)
                        & (vert_idx > int32(0))
                        & (vert_idx < nlev),
                        np.zeros_like(z_rho_e),
                        z_theta_v_e,
                    )

                if iadv_rhotheta == 2:
                    # Compute upwind-biased values for rho and theta starting from centered differences
                    # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                    # at a second-order accurate FV discretization, but twice the length is needed for numerical stability
                    (z_rho_e, z_theta_v_e) = np.where(
                        (horizontal_lower_1 < horz_idx)
                        & (horz_idx < horizontal_upper_1)
                        & (vert_idx > int32(0))
                        & (vert_idx < nlev),
                        mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1_numpy(
                            mesh=mesh,
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
                (horizontal_lower < horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx > int32(0))
                & (vert_idx < nflatlev),
                mo_solve_nonhydro_stencil_18_numpy(
                    mesh=mesh,
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=z_exner_ex_pr,
                ),
                z_gradh_exner,
            )

            if igradp_method == 3:
                # horizontal gradient of Exner pressure, including metric correction
                # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
                z_gradh_exner = np.where(
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > nflatlev)
                    & (vert_idx < (nflat_gradp + int32(1))),
                    mo_solve_nonhydro_stencil_19_numpy(
                        mesh=mesh,
                        inv_dual_edge_length=inv_dual_edge_length,
                        z_exner_ex_pr=z_exner_ex_pr,
                        ddxn_z_full=ddxn_z_full,
                        c_lin_e=c_lin_e,
                        z_dexner_dz_c_1=z_dexner_dz_c_1,
                    ),
                    z_gradh_exner,
                )

                new_shape = list(mesh.e2c.shape)
                new_shape.append(nlev)
                new_shape = tuple(new_shape)
                z_gradh_exner = np.where(
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > (nflat_gradp + int32(1)))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_20_numpy(
                        mesh=mesh,
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
            if igradp_method == 3:
                z_hydro_corr = np.where(
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > (nlev - int32(1)))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_21_numpy(
                        mesh=mesh,
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

            z_hydro_corr_horizontal = np_as_located_field(EdgeDim)(
                np.asarray(z_hydro_corr)[:, nlev - 1]
            )

            if igradp_method == 3:
                z_gradh_exner = np.where(
                    (horizontal_lower_3 < horz_idx)
                    & (horz_idx < horizontal_upper_3)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_22_numpy(
                        mesh=mesh,
                        ipeidx_dsl=ipeidx_dsl,
                        pg_exdist=pg_exdist,
                        z_hydro_corr=z_hydro_corr_horizontal,
                        z_gradh_exner=z_gradh_exner,
                    ),
                    z_gradh_exner,
                )

            vn = np.where(
                (horizontal_lower < horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx > int32(0))
                & (vert_idx < nlev),
                mo_solve_nonhydro_stencil_24_numpy(
                    mesh=mesh,
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
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_28_numpy(
                        mesh=mesh,
                        vn_incr=vn_incr,
                        vn=vn,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    vn,
                )

        else:

            z_graddiv_vn = np.where(
                (horizontal_lower_2 < horz_idx)
                & (horz_idx < horizontal_upper_2)
                & (vert_idx > kstart_dd3d)
                & (vert_idx < nlev),
                mo_solve_nonhydro_stencil_17_numpy(
                    mesh=mesh,
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
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_23_numpy(
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
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_25_numpy(
                        mesh=mesh,
                        geofac_grdiv=geofac_grdiv,
                        z_graddiv_vn=z_graddiv_vn,
                    ),
                    z_graddiv2_vn,
                )

            if lhdiff_rcf:
                if divdamp_order == 24 and scal_divdamp_o2 > 1.0e-6:
                    vn = np.where(
                        (horizontal_lower < horz_idx)
                        & (horz_idx < horizontal_upper)
                        & (vert_idx > int32(0))
                        & (vert_idx < nlev),
                        mo_solve_nonhydro_stencil_26_numpy(
                            mesh=mesh,
                            z_graddiv_vn=z_graddiv_vn,
                            vn=vn,
                            scal_divdamp_o2=scal_divdamp_o2,
                        ),
                        vn,
                    )

                if divdamp_order == 24 and divdamp_fac_o2 <= 4 * divdamp_fac:
                    if limited_area:
                        vn = np.where(
                            (horizontal_lower < horz_idx)
                            & (horz_idx < horizontal_upper)
                            & (vert_idx > int32(0))
                            & (vert_idx < nlev),
                            mo_solve_nonhydro_stencil_27_numpy(
                                mesh=mesh,
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
                            (horizontal_lower < horz_idx)
                            & (horz_idx < horizontal_upper)
                            & (vert_idx > int32(0))
                            & (vert_idx < nlev),
                            mo_solve_nonhydro_4th_order_divdamp_numpy(
                                mesh=mesh,
                                scal_divdamp=scal_divdamp,
                                z_graddiv2_vn=z_graddiv2_vn,
                                vn=vn,
                            ),
                            vn,
                        )

            if is_iau_active:
                vn = np.where(
                    (horizontal_lower < horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx > int32(0))
                    & (vert_idx < nlev),
                    mo_solve_nonhydro_stencil_28_numpy(
                        mesh=mesh,
                        vn_incr=vn_incr,
                        vn=vn,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    vn,
                )

        return z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn

    @pytest.fixture
    def input_data(self, mesh):
        geofac_grg_x = random_field(mesh, CellDim, C2E2CODim)
        geofac_grg_y = random_field(mesh, CellDim, C2E2CODim)
        p_vn = random_field(mesh, EdgeDim, KDim)
        p_vt = random_field(mesh, EdgeDim, KDim)
        pos_on_tplane_e_1 = random_field(mesh, ECDim)
        pos_on_tplane_e_2 = random_field(mesh, ECDim)
        primal_normal_cell_1 = random_field(mesh, ECDim)
        dual_normal_cell_1 = random_field(mesh, ECDim)
        primal_normal_cell_2 = random_field(mesh, ECDim)
        dual_normal_cell_2 = random_field(mesh, ECDim)
        rho_ref_me = random_field(mesh, EdgeDim, KDim)
        theta_ref_me = random_field(mesh, EdgeDim, KDim)
        z_rth_pr_1 = random_field(mesh, CellDim, KDim)
        z_rth_pr_2 = random_field(mesh, CellDim, KDim)
        ddxn_z_full = random_field(mesh, EdgeDim, KDim)
        c_lin_e = random_field(mesh, EdgeDim, E2CDim)
        z_exner_ex_pr = random_field(mesh, CellDim, KDim)
        z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim)
        z_dexner_dz_c_2 = random_field(mesh, CellDim, KDim)
        z_hydro_corr = random_field(mesh, EdgeDim, KDim)
        theta_v = random_field(mesh, CellDim, KDim)
        zdiff_gradp = random_field(mesh, ECDim, KDim)
        theta_v_ic = random_field(mesh, CellDim, KDim)
        inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
        ipeidx_dsl = random_mask(mesh, EdgeDim, KDim)
        pg_exdist = random_field(mesh, EdgeDim, KDim)
        hmask_dd3d = random_field(mesh, EdgeDim)
        scalfac_dd3d = random_field(mesh, KDim)
        z_dwdz_dd = random_field(mesh, CellDim, KDim)
        inv_dual_edge_length = random_field(mesh, EdgeDim)
        ddt_vn_apc_ntl2 = random_field(mesh, EdgeDim, KDim)
        vn_nnow = random_field(mesh, EdgeDim, KDim)
        ddt_vn_apc_ntl1 = random_field(mesh, EdgeDim, KDim)
        ddt_vn_phy = random_field(mesh, EdgeDim, KDim)
        z_graddiv_vn = random_field(mesh, EdgeDim, KDim)
        vn_incr = random_field(mesh, EdgeDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)
        z_theta_v_e = random_field(mesh, EdgeDim, KDim)
        z_gradh_exner = random_field(mesh, EdgeDim, KDim)
        z_graddiv2_vn = random_field(mesh, EdgeDim, KDim)
        z_rho_e = random_field(mesh, EdgeDim, KDim)
        geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
        scal_divdamp = random_field(mesh, KDim)
        bdy_divdamp = random_field(mesh, KDim)
        nudgecoeff_e = random_field(mesh, EdgeDim)

        ikoffset = zero_field(mesh, EdgeDim, E2CDim, KDim, dtype=int32)
        rng = np.random.default_rng()

        for k in range(mesh.k_level):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=mesh.k_level - k - 1,
                size=(ikoffset.shape[0], ikoffset.shape[1]),
            )
        ikoffset = flatten_first_two_dims(ECDim, KDim, field=ikoffset)

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        horz_idx = zero_field(mesh, EdgeDim, dtype=int32)
        for edge in range(mesh.n_edges):
            horz_idx[edge] = edge

        grav_o_cpd = 9.80665 / 1004.64
        dtime = 0.9
        p_dthalf = 0.5 * dtime
        idiv_method = 1
        igradp_method = 3
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
        horizontal_lower = 5387
        horizontal_upper = 31558
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
        nlev = mesh.k_level
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
            z_graddiv_vn=z_graddiv_vn,
            vn_incr=vn_incr,
            vn=vn,
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            z_graddiv2_vn=z_graddiv2_vn,
            z_hydro_corr=z_hydro_corr,
            geofac_grdiv=geofac_grdiv,
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            grav_o_cpd=grav_o_cpd,
            p_dthalf=p_dthalf,
            idiv_method=idiv_method,
            igradp_method=igradp_method,
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
            horizontal_lower=horizontal_lower,
            horizontal_upper=horizontal_upper,
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
            nlev=nlev,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
