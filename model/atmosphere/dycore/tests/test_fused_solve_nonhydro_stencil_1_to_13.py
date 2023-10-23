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

from icon4py.model.atmosphere.dycore.fused_mo_solve_nonhydro_stencils_01_to_13 import (
    fused_mo_solve_nonhydro_stencils_01_to_13,
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


class TestFusedMoSolveNonHydroStencil1To13(StencilTest):
    PROGRAM = fused_mo_solve_nonhydro_stencils_01_to_13
    OUTPUTS = ("z_rho_e", "z_theta_v_e", "z_gradh_exner", "vn", "z_graddiv_vn")

    # flake8: noqa: C901
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



        return dict(
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn=vn,
            z_graddiv_vn=z_graddiv_vn,
        )

    @pytest.fixture
    def input_data(self, mesh):

        rho = random_field(mesh, CellDim, KDim)
        rho_ref_mc = random_field(mesh, CellDim, KDim)
        theta_v = random_field(mesh, CellDim, KDim)
        theta_ref_mc = random_field(mesh, CellDim, KDim)
        z_rth_pr_1 = random_field(mesh, CellDim, KDim)
        z_rth_pr_2 = random_field(mesh, CellDim, KDim)
        z_theta_v_pr_ic = random_field(mesh, CellDim, KDim)
        theta_ref_ic = random_field(mesh, CellDim, KDim)

        d2dexdz2_fac1_mc = random_field(mesh, CellDim, KDim)
        d2dexdz2_fac2_mc = random_field(mesh, CellDim, KDim)
        wgtfacq_c_dsl = random_field(mesh, CellDim, KDim)
        wgtfac_c = random_field(mesh, CellDim, KDim)
        vwind_expl_wgt = random_field(mesh, CellDim, KDim)
        exner_pr = random_field(mesh, CellDim, KDim)
        d_exner_dz_ref_ic = random_field(mesh, CellDim, KDim)
        ddqz_z_half = random_field(mesh, CellDim, KDim)

        z_th_ddz_exner_c = random_field(mesh, CellDim, KDim)
        rho_ic = random_field(mesh, CellDim, KDim)
        z_exner_ic = random_field(mesh, CellDim, KDim)
        exner_exfac = random_field(mesh, CellDim, KDim)
        exner = random_field(mesh, CellDim, KDim)
        exner_ref_mc = random_field(mesh, CellDim, KDim)
        z_exner_ex_pr = random_field(mesh, CellDim, KDim)

        z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim)
        z_dexner_dz_c_2 = random_field(mesh, CellDim, KDim)

        theta_v_ic = random_field(mesh, CellDim, KDim)
        inv_ddqz_z_full = random_field(mesh, CellDim, KDim)

        w_concorr_c = random_field(mesh, CellDim, KDim)
        w = random_field(mesh, CellDim, KDim)
        rho_now = random_field(mesh, CellDim, KDim)
        rho_var = random_field(mesh, CellDim, KDim)
        theta_now = random_field(mesh, CellDim, KDim)
        theta_var = random_field(mesh, CellDim, KDim)

        k_field = random_field(mesh, KDim)
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
        istep = 1
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
            rho=rho,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_ref_ic=theta_ref_ic,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            wgtfacq_c_dsl=wgtfacq_c_dsl,
            wgtfac_c=wgtfac_c,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            rho_ic=rho_ic,
            z_exner_ic=z_exner_ic,
            exner_exfac=exner_exfac,
            exner=exner,
            exner_ref_mc=exner_ref_mc,
            z_exner_ex_pr=z_exner_ex_pr,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            limited_area=limited_area,
            igradp_method=igradp_method,
            k_field=k_field,
            w=w,
            w_concorr_c=w_concorr_c,
            rho_now=rho_now,
            rho_var=rho_var,
            theta_now=theta_now,
            theta_var=theta_var,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
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
