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
    OUTPUTS = (
        "z_exner_ex_pr",
        "exner_pr",
        "z_exner_ic",
        "vn",
        "z_dexner_dz_c_1",
        "z_rth_pr_1",
        "z_rth_pr_2",
        "rho_ic",
        "z_theta_v_pr_ic",
        "theta_v_ic",
        "z_th_ddz_exner_c",
        "z_dexner_dz_c_2",
    )

    # flake8: noqa: C901
    @classmethod
    def reference(
        cls,
        mesh,
        rho,
        rho_ref_mc,
        theta_v,
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
        exner,
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
        k_field,
        w,
        w_concorr_c,
        rho_now,
        rho_var,
        theta_now,
        theta_var,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        istep,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        horizontal_lower_01,
        horizontal_upper_01,
        horizontal_lower_02,
        horizontal_upper_02,
        horizontal_lower_03,
        horizontal_upper_03,
        horizontal_lower_11,
        horizontal_upper_11,
        nlev,
        nflatlev,
        nflat_gradp,
    ):
        iadv_rhotheta = 2
        horz_idx = horz_idx[:, np.newaxis]
        z_grad_rth_1 = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_grad_rth_2 = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        exner_pr = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_exner_ic = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        rho_ic = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_theta_v_pr_ic = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))
        z_th_ddz_exner_c = np_as_located_field(CellDim, KDim)(np.zeros_like(z_rth_pr_1))

        return dict(
            z_exner_ex_pr=z_exner_ex_pr,
            exner_pr=exner_pr,
            z_exner_ic=z_exner_ic,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
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
        vwind_expl_wgt = random_field(mesh, CellDim)
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

        k_field = zero_field(mesh, KDim, dtype=int32)  # TODO: @abishekg7 change later

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        horz_idx = zero_field(mesh, CellDim, dtype=int32)
        for cell in range(mesh.n_cells):
            horz_idx[cell] = cell

        grav_o_cpd = 9.80665 / 1004.64
        dtime = 0.9
        p_dthalf = 0.5 * dtime
        idiv_method = 1
        igradp_method = 3
        wgt_nnow_rth = 0.25
        wgt_nnew_rth = 0.75
        cpd = 1004.64
        iau_wgt_dyn = 1.0
        is_iau_active = False
        limited_area = True
        itime_scheme = 4
        istep = 1
        horizontal_lower = 5387
        horizontal_upper = 31558
        horizontal_lower_01 = 31558
        horizontal_upper_01 = 31558
        horizontal_lower_02 = 31558
        horizontal_upper_02 = 31558
        horizontal_lower_03 = 3777
        horizontal_upper_03 = 31558
        horizontal_lower_11 = 3777
        horizontal_upper_11 = 31558

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
            horizontal_start=horizontal_lower,
            horizontal_end=horizontal_upper,
            vertical_start=0,
            vertical_end=nlev,
            horizontal_lower_01=horizontal_lower_01,
            horizontal_upper_01=horizontal_upper_01,
            horizontal_lower_02=horizontal_lower_02,
            horizontal_upper_02=horizontal_upper_02,
            horizontal_lower_03=horizontal_lower_03,
            horizontal_upper_03=horizontal_upper_03,
            horizontal_lower_11=horizontal_lower_11,
            horizontal_upper_11=horizontal_upper_11,
            nlev=nlev,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
