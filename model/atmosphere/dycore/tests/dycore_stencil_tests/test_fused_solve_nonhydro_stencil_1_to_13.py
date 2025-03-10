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

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_first_vertical_derivative import (
    compute_first_vertical_derivative_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_virtual_potential_temperatures_and_pressure_gradient import (
    compute_virtual_potential_temperatures_and_pressure_gradient_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_extrapolate_temporally_exner_pressure import (
    extrapolate_temporally_exner_pressure_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_interpolate_to_half_levels_vp import (
    interpolate_to_half_levels_vp_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_interpolate_to_surface import (
    interpolate_to_surface_numpy,
)
from model.atmosphere.dycore.tests.dycore_stencil_tests.test_set_theta_v_prime_ic_at_lower_boundary import (
    set_theta_v_prime_ic_at_lower_boundary_numpy,
)

from icon4py.model.atmosphere.dycore.fused_mo_solve_nonhydro_stencils_1_to_13 import (
    fused_mo_solve_nonhydro_stencils_01_to_13,
)
from icon4py.model.common.dimension import (
    CellDim,
    KDim,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils.helpers import StencilTest
from icon4py.model.common.utils import data_allocation as data_alloc


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
        rho_nnow: np.ndarray,
        rho_ref_mc: np.ndarray,
        theta_v_nnow: np.ndarray,
        theta_ref_mc: np.ndarray,
        z_rth_pr_1: np.ndarray,
        z_rth_pr_2: np.ndarray,
        z_theta_v_pr_ic: np.ndarray,
        theta_ref_ic: np.ndarray,
        d2dexdz2_fac1_mc: np.ndarray,
        d2dexdz2_fac2_mc: np.ndarray,
        wgtfacq_c_dsl: np.ndarray,
        wgtfac_c: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        exner_pr: np.ndarray,
        d_exner_dz_ref_ic: np.ndarray,
        ddqz_z_half: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        rho_ic: np.ndarray,
        z_exner_ic: np.ndarray,
        exner_exfac: np.ndarray,
        exner_nnow: np.ndarray,
        exner_ref_mc: np.ndarray,
        z_exner_ex_pr: np.ndarray,
        z_dexner_dz_c_1: np.ndarray,
        z_dexner_dz_c_2: np.ndarray,
        theta_v_ic: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        horz_idx: np.ndarray,
        vert_idx: np.ndarray,
        limited_area: np.ndarray,
        igradp_method: np.ndarray,
        w: np.ndarray,
        w_concorr_c: np.ndarray,
        rho_nvar: np.ndarray,
        theta_v_nvar: np.ndarray,
        dtime: float,
        wgt_nnow_rth: float,
        wgt_nnew_rth: float,
        istep: int,
        start_cell_lateral_boundary: int,
        start_cell_lateral_boundary_level_3: int,
        end_cell_halo: int,
        end_cell_end: int,
        start_cell_halo_level_2: int,
        end_cell_halo_level_2: int,
        end_cell_local: int,
        n_lev: int,
        nflatlev: int,
        nflat_gradp: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ):
        horz_idx = horz_idx[:, np.newaxis]

        if istep == 1:
            if limited_area:
                (z_rth_pr_1, z_rth_pr_2) = np.where(
                    (start_cell_lateral_boundary <= horz_idx) & (horz_idx < end_cell_end),
                    (0.0, 0.0),
                    (z_rth_pr_1, z_rth_pr_2),
                )

            (z_exner_ex_pr, exner_pr) = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx < n_lev),
                extrapolate_temporally_exner_pressure_numpy(
                    grid,
                    exner_exfac=exner_exfac,
                    exner=exner_nnow,
                    exner_ref_mc=exner_ref_mc,
                    exner_pr=exner_pr,
                ),
                (z_exner_ex_pr, exner_pr),
            )

            z_exner_ex_pr = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx == n_lev),
                0.0,
                z_exner_ex_pr,
            )

            vert_start = np.maximum(1, nflatlev)
            if igradp_method == 3:
                z_exner_ic[:, :n_lev] = np.where(
                    (start_cell_lateral_boundary_level_3 <= horz_idx)
                    & (horz_idx < end_cell_halo)
                    & (vert_idx == n_lev),
                    interpolate_to_surface_numpy(
                        interpolant=z_exner_ex_pr,
                        wgtfacq_c=wgtfacq_c_dsl,
                        interpolation_to_surface=z_exner_ic[:, :n_lev],
                    ),
                    z_exner_ic[:, :n_lev],
                )
                z_exner_ic[:, :n_lev] = np.where(
                    (start_cell_lateral_boundary_level_3 <= horz_idx) & (horz_idx < end_cell_halo),
                    interpolate_to_half_levels_vp_numpy(
                        wgtfac_c=wgtfac_c, interpolant=z_exner_ex_pr
                    ),
                    z_exner_ic[:, :n_lev],
                )

                z_dexner_dz_c_1 = np.where(
                    (start_cell_lateral_boundary_level_3 <= horz_idx)
                    & (horz_idx < end_cell_halo)
                    & (vert_idx < (n_lev + int32(1))),
                    compute_first_vertical_derivative_numpy(
                        z_exner_ic=z_exner_ic, inv_ddqz_z_full=inv_ddqz_z_full
                    ),
                    z_dexner_dz_c_1,
                )

            (z_rth_pr_1, z_rth_pr_2) = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx == int32(0)),
                compute_perturbation_of_rho_and_theta_numpy(
                    rho=rho_nnow,
                    rho_ref_mc=rho_ref_mc,
                    theta_v=theta_v_nnow,
                    theta_ref_mc=theta_ref_mc,
                ),
                (z_rth_pr_1, z_rth_pr_2),
            )

            (rho_ic, z_rth_pr_1, z_rth_pr_2) = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx >= int32(1)),
                compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy(
                    wgtfac_c=wgtfac_c,
                    rho=rho_nnow,
                    rho_ref_mc=rho_ref_mc,
                    theta_v=theta_v_nnow,
                    theta_ref_mc=theta_ref_mc,
                ),
                (rho_ic, z_rth_pr_1, z_rth_pr_2),
            )

            (z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c) = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx >= int32(1)),
                compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
                    wgtfac_c=wgtfac_c,
                    z_rth_pr_2=z_rth_pr_2,
                    theta_v=theta_v_nnow,
                    vwind_expl_wgt=vwind_expl_wgt,
                    exner_pr=exner_pr,
                    d_exner_dz_ref_ic=d_exner_dz_ref_ic,
                    ddqz_z_half=ddqz_z_half,
                ),
                (z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c),
            )

            z_theta_v_pr_ic[:, 0] = 0.0

            (z_theta_v_pr_ic[:, :n_lev], theta_v_ic) = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx == n_lev),
                set_theta_v_prime_ic_at_lower_boundary_numpy(
                    wgtfacq_c=wgtfacq_c_dsl,
                    z_rth_pr=z_rth_pr_2,
                    theta_ref_ic=theta_ref_ic,
                    z_theta_v_pr_ic=z_theta_v_pr_ic[:, :n_lev],
                ),
                (z_theta_v_pr_ic[:, :n_lev], theta_v_ic),
            )

            vert_start = nflat_gradp
            if igradp_method == 3:
                z_dexner_dz_c_2 = np.where(
                    (start_cell_lateral_boundary_level_3 <= horz_idx)
                    & (horz_idx < end_cell_halo)
                    & (vert_start <= vert_idx),
                    compute_approx_of_2nd_vertical_derivative_of_exner_numpy(
                        z_theta_v_pr_ic=z_theta_v_pr_ic,
                        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
                        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
                        z_rth_pr_2=z_rth_pr_2,
                    ),
                    z_dexner_dz_c_2,
                )

            (z_rth_pr_1, z_rth_pr_2) = np.where(
                (start_cell_halo_level_2 <= horz_idx) & (horz_idx < end_cell_halo_level_2),
                compute_perturbation_of_rho_and_theta_numpy(
                    rho=rho_nnow,
                    rho_ref_mc=rho_ref_mc,
                    theta_v=theta_v_nnow,
                    theta_ref_mc=theta_ref_mc,
                ),
                (z_rth_pr_1, z_rth_pr_2),
            )

        else:
            (rho_ic, z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c) = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_local)
                & (int32(1) <= vert_idx),
                compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy(
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
                (rho_ic, z_theta_v_pr_ic[:, :n_lev], theta_v_ic, z_th_ddz_exner_c),
            )

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
        rho_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        theta_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        z_rth_pr_1 = data_alloc.random_field(grid, CellDim, KDim)
        z_rth_pr_2 = data_alloc.random_field(grid, CellDim, KDim)
        z_theta_v_pr_ic = data_alloc.random_field(grid, CellDim, KDim, extend={KDim: 1})
        theta_ref_ic = data_alloc.random_field(grid, CellDim, KDim)
        d2dexdz2_fac1_mc = data_alloc.random_field(grid, CellDim, KDim)
        d2dexdz2_fac2_mc = data_alloc.random_field(grid, CellDim, KDim)
        wgtfacq_c_dsl = data_alloc.random_field(grid, CellDim, KDim)
        wgtfac_c = data_alloc.random_field(grid, CellDim, KDim)
        vwind_expl_wgt = data_alloc.random_field(grid, CellDim)
        exner_pr = data_alloc.random_field(grid, CellDim, KDim)
        d_exner_dz_ref_ic = data_alloc.random_field(grid, CellDim, KDim)
        ddqz_z_half = data_alloc.random_field(grid, CellDim, KDim)
        z_th_ddz_exner_c = data_alloc.random_field(grid, CellDim, KDim)
        rho_ic = data_alloc.random_field(grid, CellDim, KDim)
        z_exner_ic = data_alloc.random_field(grid, CellDim, KDim, extend={KDim: 1})
        exner_exfac = data_alloc.random_field(grid, CellDim, KDim)
        exner_nnow = data_alloc.random_field(grid, CellDim, KDim)
        exner_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        z_exner_ex_pr = data_alloc.random_field(grid, CellDim, KDim)
        z_dexner_dz_c_1 = data_alloc.random_field(grid, CellDim, KDim)
        z_dexner_dz_c_2 = data_alloc.random_field(grid, CellDim, KDim)
        theta_v_ic = data_alloc.random_field(grid, CellDim, KDim)
        inv_ddqz_z_full = data_alloc.random_field(grid, CellDim, KDim)
        w_concorr_c = data_alloc.random_field(grid, CellDim, KDim)
        w = data_alloc.random_field(grid, CellDim, KDim)
        rho_nnow = data_alloc.random_field(grid, CellDim, KDim)
        rho_nvar = data_alloc.random_field(grid, CellDim, KDim)
        theta_v_nnow = data_alloc.random_field(grid, CellDim, KDim)
        theta_v_nvar = data_alloc.random_field(grid, CellDim, KDim)

        vert_idx = data_alloc.index_field(grid, KDim)
        horz_idx = data_alloc.index_field(grid, CellDim)

        dtime = 0.9
        igradp_method = 3
        wgt_nnow_rth = 0.25
        wgt_nnew_rth = 0.75
        limited_area = True
        istep = 1

        cell_domain = h_grid.domain(CellDim)
        n_lev = grid.num_levels
        start_cell_lateral_boundary = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
        start_cell_lateral_boundary_level_3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        end_cell_halo = grid.end_index(cell_domain(h_grid.Zone.HALO))
        end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
        start_cell_halo_level_2 = grid.start_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
        end_cell_halo_level_2 = grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        horizontal_start = 0
        horizontal_end = grid.num_cells
        vertical_start = 0
        vertical_end = n_lev

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
            start_cell_lateral_boundary=start_cell_lateral_boundary,
            start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
            end_cell_halo=end_cell_halo,
            end_cell_end=end_cell_end,
            start_cell_halo_level_2=start_cell_halo_level_2,
            end_cell_halo_level_2=end_cell_halo_level_2,
            end_cell_local=end_cell_local,
            n_lev=n_lev,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
