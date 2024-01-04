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
from .test_mo_solve_nonhydro_stencil_01 import mo_solve_nonhydro_stencil_01_numpy
from .test_mo_solve_nonhydro_stencil_02 import mo_solve_nonhydro_stencil_02_numpy
from .test_mo_solve_nonhydro_stencil_04 import mo_solve_nonhydro_stencil_04_numpy
from .test_mo_solve_nonhydro_stencil_05 import mo_solve_nonhydro_stencil_05_numpy
from .test_mo_solve_nonhydro_stencil_06 import mo_solve_nonhydro_stencil_06_numpy
from .test_mo_solve_nonhydro_stencil_07 import mo_solve_nonhydro_stencil_07_numpy
from .test_mo_solve_nonhydro_stencil_08 import mo_solve_nonhydro_stencil_08_numpy
from .test_mo_solve_nonhydro_stencil_09 import mo_solve_nonhydro_stencil_09_numpy
from .test_mo_solve_nonhydro_stencil_10 import mo_solve_nonhydro_stencil_10_numpy
from .test_mo_solve_nonhydro_stencil_11_lower import (
    mo_solve_nonhydro_stencil_11_lower_numpy,
)
from .test_mo_solve_nonhydro_stencil_11_upper import (
    mo_solve_nonhydro_stencil_11_upper_numpy,
)
from .test_mo_solve_nonhydro_stencil_12 import mo_solve_nonhydro_stencil_12_numpy
from .test_mo_solve_nonhydro_stencil_13 import mo_solve_nonhydro_stencil_13_numpy

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
        grid,
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
        n_lev,
        nflatlev,
        nflat_gradp,
    ):
        horz_idx = horz_idx[:, np.newaxis]

        if istep == 1:
            if limited_area:
                (z_rth_pr_1, z_rth_pr_2) = mo_solve_nonhydro_stencil_01_numpy(
                                            grid,
                                            z_rth_pr_1=z_rth_pr_1,
                                            z_rth_pr_2=z_rth_pr_2,
                                            )


            (z_exner_ex_pr, exner_pr) = np.where(
                (horizontal_lower_02 <= horz_idx)
                & (horz_idx < horizontal_upper_02),
                mo_solve_nonhydro_stencil_02_numpy(
                    grid,
                    exner_exfac=exner_exfac,
                    exner=exner_nnow,
                    exner_ref_mc=exner_ref_mc,
                    exner_pr=exner_pr,
                ),
                (z_exner_ex_pr, exner_pr),
            )

            z_exner_ex_pr = np.where(
                (horizontal_lower_02 <= horz_idx)
                & (horz_idx < horizontal_upper_02)
                & (vert_idx == n_lev),
                0.0,
                z_exner_ex_pr,
            )

            vert_start = np.maximum(1, nflatlev)
            if igradp_method == 3:
                z_exner_ic[:, :n_lev] = np.where(
                    (horizontal_lower_02 <= horz_idx)
                    & (horz_idx < horizontal_upper_02)
                    & (vert_idx == n_lev),
                    mo_solve_nonhydro_stencil_04_numpy(
                        grid,
                        z_exner_ex_pr=z_exner_ex_pr,
                        wgtfacq_c=wgtfacq_c_dsl,
                        z_exner_ic=z_exner_ic[:, :n_lev],
                    ),
                    z_exner_ic[:, :n_lev],
                )
                z_exner_ic[:, :n_lev] = np.where(
                    (horizontal_lower_02 <= horz_idx)
                    & (horz_idx < horizontal_upper_02)
                    & (vert_start <= vert_idx)
                    & (vert_idx < (n_lev + int32(1))),
                    mo_solve_nonhydro_stencil_05_numpy(
                        grid, wgtfac_c=wgtfac_c, z_exner_ex_pr=z_exner_ex_pr
                    ),
                    z_exner_ic[:, :n_lev],
                )

                z_dexner_dz_c_1 = np.where(
                    (horizontal_lower_02 <= horz_idx)
                    & (horz_idx < horizontal_upper_02)
                    & (vert_start <= vert_idx)
                    & (vert_idx < (n_lev + int32(1))),
                    mo_solve_nonhydro_stencil_06_numpy(
                        grid, z_exner_ic=z_exner_ic, inv_ddqz_z_full=inv_ddqz_z_full
                    ),
                    z_dexner_dz_c_1,
                )

        #     (z_rth_pr_1, z_rth_pr_2) = np.where(
        #         (horizontal_lower_02 <= horz_idx)
        #         & (horz_idx < horizontal_upper_02)
        #         & (vert_idx == int32(0)),
        #         mo_solve_nonhydro_stencil_07_numpy(
        #             mesh=mesh,
        #             rho=rho_nnow,
        #             rho_ref_mc=rho_ref_mc,
        #             theta_v=theta_v_nnow,
        #             theta_ref_mc=theta_ref_mc,
        #         ),
        #         (z_rth_pr_1, z_rth_pr_2),
        #     )
        #
        #     (rho_ic, z_rth_pr_1, z_rth_pr_2) = np.where(
        #         (horizontal_lower_02 <= horz_idx)
        #         & (horz_idx < horizontal_upper_02)
        #         & (vert_idx >= int32(0)),
        #         mo_solve_nonhydro_stencil_08_numpy(
        #             mesh=mesh,
        #             wgtfac_c=wgtfac_c,
        #             rho=rho_nnow,
        #             rho_ref_mc=rho_ref_mc,
        #             theta_v=theta_v_nnow,
        #             theta_ref_mc=theta_ref_mc,
        #         ),
        #         (rho_ic, z_rth_pr_1, z_rth_pr_2),
        #     )
        #
        #     (z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c) = np.where(
        #         (horizontal_lower_02 <= horz_idx)
        #         & (horz_idx < horizontal_upper_02)
        #         & (vert_idx >= int32(0)),
        #         mo_solve_nonhydro_stencil_09_numpy(
        #             mesh=mesh,
        #             wgtfac_c=wgtfac_c,
        #             z_rth_pr_2=z_rth_pr_2,
        #             theta_v=theta_v_nnow,
        #             vwind_expl_wgt=vwind_expl_wgt,
        #             exner_pr=exner_pr,
        #             d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        #             ddqz_z_half=ddqz_z_half,
        #         ),
        #         (z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c),
        #     )
        #
        #     z_theta_v_pr_ic[:, :n_lev] = np.where(
        #         (horizontal_lower_02 <= horz_idx)
        #         & (horz_idx < horizontal_upper_02)
        #         & (vert_idx == int32(0)),
        #         mo_solve_nonhydro_stencil_11_lower_numpy(
        #             mesh=mesh, z_theta_v_pr_ic=z_theta_v_pr_ic[:, :n_lev]
        #         ),
        #         z_theta_v_pr_ic[:, :n_lev],
        #     )
        #
        #     (z_theta_v_pr_ic[:, :n_lev], theta_v_ic) = np.where(
        #         (horizontal_lower_02 <= horz_idx)
        #         & (horz_idx < horizontal_upper_02)
        #         & (vert_idx == n_lev),
        #         mo_solve_nonhydro_stencil_11_upper_numpy(
        #             mesh=mesh,
        #             wgtfacq_c=wgtfacq_c_dsl,
        #             z_rth_pr=z_rth_pr_2,
        #             theta_ref_ic=theta_ref_ic,
        #             z_theta_v_pr_ic=z_theta_v_pr_ic[:, :n_lev],
        #         ),
        #         (z_theta_v_pr_ic[:, :n_lev], theta_v_ic),
        #     )
        #
        #     vert_start = nflat_gradp
        #     if igradp_method == 3:
        #         z_dexner_dz_c_2 = np.where(
        #             (horizontal_lower_02 <= horz_idx)
        #             & (horz_idx < horizontal_upper_02)
        #             & (vert_start <= vert_idx),
        #             mo_solve_nonhydro_stencil_12_numpy(
        #                 mesh=mesh,
        #                 z_theta_v_pr_ic=z_theta_v_pr_ic,
        #                 d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        #                 d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        #                 z_rth_pr_2=z_rth_pr_2,
        #             ),
        #             z_dexner_dz_c_2,
        #         )
        #
        #     (z_rth_pr_1, z_rth_pr_2) = np.where(
        #         (horizontal_lower_03 <= horz_idx) & (horz_idx < horizontal_upper_03),
        #         mo_solve_nonhydro_stencil_13_numpy(
        #             mesh=mesh,
        #             rho=rho_nnow,
        #             rho_ref_mc=rho_ref_mc,
        #             theta_v=theta_v_nnow,
        #             theta_ref_mc=theta_ref_mc,
        #         ),
        #         (z_rth_pr_1, z_rth_pr_2),
        #     )
        #
        # else:
        #     (rho_ic, z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c) = np.where(
        #         (horizontal_lower_11 <= horz_idx)
        #         & (horz_idx < horizontal_upper_11)
        #         & (int32(1) <= vert_idx),
        #         mo_solve_nonhydro_stencil_10_numpy(
        #             mesh=mesh,
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
        #         (rho_ic, z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c),
        #     )

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
    def input_data(self, grid):
        rho_ref_mc = random_field(grid, CellDim, KDim)
        theta_ref_mc = random_field(grid, CellDim, KDim)
        z_rth_pr_1 = random_field(grid, CellDim, KDim)
        z_rth_pr_2 = random_field(grid, CellDim, KDim)
        z_theta_v_pr_ic = random_field(grid, CellDim, KDim, extend={KDim: 1})
        theta_ref_ic = random_field(grid, CellDim, KDim)
        d2dexdz2_fac1_mc = random_field(grid, CellDim, KDim)
        d2dexdz2_fac2_mc = random_field(grid, CellDim, KDim)
        wgtfacq_c_dsl = random_field(grid, CellDim, KDim)
        wgtfac_c = random_field(grid, CellDim, KDim)
        vwind_expl_wgt = random_field(grid, CellDim)
        exner_pr = random_field(grid, CellDim, KDim)
        d_exner_dz_ref_ic = random_field(grid, CellDim, KDim)
        ddqz_z_half = random_field(grid, CellDim, KDim)
        z_th_ddz_exner_c = random_field(grid, CellDim, KDim)
        rho_ic = random_field(grid, CellDim, KDim)
        z_exner_ic = random_field(grid, CellDim, KDim, extend={KDim: 1})
        exner_exfac = random_field(grid, CellDim, KDim)
        exner_nnow = random_field(grid, CellDim, KDim)
        exner_ref_mc = random_field(grid, CellDim, KDim)
        z_exner_ex_pr = random_field(grid, CellDim, KDim)
        z_dexner_dz_c_1 = random_field(grid, CellDim, KDim)
        z_dexner_dz_c_2 = random_field(grid, CellDim, KDim)
        theta_v_ic = random_field(grid, CellDim, KDim)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim)
        w_concorr_c = random_field(grid, CellDim, KDim)
        w = random_field(grid, CellDim, KDim)
        rho_nnow = random_field(grid, CellDim, KDim)
        rho_nvar = random_field(grid, CellDim, KDim)
        theta_v_nnow = random_field(grid, CellDim, KDim)
        theta_v_nvar = random_field(grid, CellDim, KDim)

        vert_idx = zero_field(grid, KDim, dtype=int32)
        for level in range(grid.num_levels):
            vert_idx[level] = level

        horz_idx = zero_field(grid, CellDim, dtype=int32)
        for cell in range(grid.num_cells):
            horz_idx[cell] = cell

        dtime = 0.9
        igradp_method = 3
        wgt_nnow_rth = 0.25
        wgt_nnew_rth = 0.75
        limited_area = True
        istep = 1
        n_lev = grid.num_levels
        horizontal_start = 0
        horizontal_end = grid.num_cells
        vertical_start = 0
        vertical_end = n_lev
        horizontal_lower_01 = horizontal_start
        horizontal_upper_01 = horizontal_end
        horizontal_lower_02 = 4
        horizontal_upper_02 = horizontal_end
        horizontal_lower_03 = horizontal_end
        horizontal_upper_03 = horizontal_end
        horizontal_lower_11 = 4
        horizontal_upper_11 = horizontal_end

        nflatlev = 4
        nflat_gradp = 27

        return dict(
            rho_nnow=rho_nnow,
            rho_ref_mc=rho_ref_mc,
            theta_v_nnow=theta_v_nnow,
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
            exner_nnow=exner_nnow,
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
            w=w,
            w_concorr_c=w_concorr_c,
            rho_nvar=rho_nvar,
            theta_v_nvar=theta_v_nvar,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
            istep=istep,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
            horizontal_lower_01=horizontal_lower_01,
            horizontal_upper_01=horizontal_upper_01,
            horizontal_lower_02=horizontal_lower_02,
            horizontal_upper_02=horizontal_upper_02,
            horizontal_lower_03=horizontal_lower_03,
            horizontal_upper_03=horizontal_upper_03,
            horizontal_lower_11=horizontal_lower_11,
            horizontal_upper_11=horizontal_upper_11,
            n_lev=n_lev,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
