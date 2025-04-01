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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.fused_mo_solve_nonhydro_stencils_1_to_13 import (
    fused_mo_solve_nonhydro_stencils_1_to_13_corrector,
    fused_mo_solve_nonhydro_stencils_1_to_13_predictor,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_first_vertical_derivative import (
    compute_first_vertical_derivative_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_compute_virtual_potential_temperatures_and_pressure_gradient import (
    compute_virtual_potential_temperatures_and_pressure_gradient_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_extrapolate_temporally_exner_pressure import (
    extrapolate_temporally_exner_pressure_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_interpolate_cell_field_to_half_levels_vp import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_interpolate_to_surface import (
    interpolate_to_surface_numpy,
)
from icon4py.model.atmosphere.dycore.tests.dycore_stencil_tests.test_set_theta_v_prime_ic_at_lower_boundary import (
    set_theta_v_prime_ic_at_lower_boundary_numpy,
)
from icon4py.model.common import type_alias as ta

# TODO
from icon4py.model.common.dimension import (
    CellDim,
    KDim,
)
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers


class TestFusedMoSolveNonHydroStencil1To13Predictor(helpers.StencilTest):
    PROGRAM = fused_mo_solve_nonhydro_stencils_1_to_13_predictor
    OUTPUTS = (
        "z_exner_ex_pr",
        "exner_pr",
        "z_exner_ic",
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
        connectivities: dict[gtx.Dimension, np.ndarray],
        rho_nnow: np.ndarray,
        rho_ref_mc: np.ndarray,
        theta_v_nnow: np.ndarray,
        theta_ref_mc: np.ndarray,
        z_rth_pr_1: np.ndarray,
        z_rth_pr_2: np.ndarray,
        z_theta_v_pr_ic: np.ndarray,
        theta_ref_ic: np.ndarray,
        wgtfacq_c: np.ndarray,
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
        d2dexdz2_fac1_mc: np.ndarray,
        d2dexdz2_fac2_mc: np.ndarray,
        horz_idx: np.ndarray,
        vert_idx: np.ndarray,
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
    ) -> dict:
        horz_idx = horz_idx[:, np.newaxis]

        # if istep == 1:
        if limited_area:
            (z_rth_pr_1, z_rth_pr_2[:, :n_lev]) = np.where(
                (start_cell_lateral_boundary <= horz_idx) & (horz_idx < end_cell_end),
                (np.zeros_like(z_rth_pr_1), np.zeros_like(z_rth_pr_2[:, :n_lev])),
                (z_rth_pr_1, z_rth_pr_2[:, :n_lev]),
            )

        (z_exner_ex_pr[:, :n_lev], exner_pr) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx) & (horz_idx < end_cell_halo),
            extrapolate_temporally_exner_pressure_numpy(
                connectivities=connectivities,
                exner=exner_nnow,
                exner_ref_mc=exner_ref_mc,
                exner_pr=exner_pr,
                exner_exfac=exner_exfac,
            ),
            (z_exner_ex_pr[:, :n_lev], exner_pr),
        )

        z_exner_ex_pr = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx == n_lev),
            np.zeros_like(z_exner_ex_pr),
            z_exner_ex_pr,
        )

        if igradp_method == 3:
            z_exner_ic = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx == n_lev),
                interpolate_to_surface_numpy(
                    interpolant=z_exner_ex_pr,
                    wgtfacq_c=wgtfacq_c,
                    interpolation_to_surface=z_exner_ic,
                ),
                z_exner_ic,
            )
            z_exner_ic = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (max(1, nflatlev) <= vert_idx)
                & (vert_idx < n_lev),
                interpolate_cell_field_to_half_levels_vp_numpy(
                    wgtfac_c=wgtfac_c, interpolant=z_exner_ex_pr
                ),
                z_exner_ic,
            )

            z_dexner_dz_c_1 = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (nflatlev <= vert_idx[:n_lev]),
                compute_first_vertical_derivative_numpy(
                    z_exner_ic=z_exner_ic, inv_ddqz_z_full=inv_ddqz_z_full
                ),
                z_dexner_dz_c_1,
            )

        (z_rth_pr_1, z_rth_pr_2[:, :n_lev]) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx[:n_lev] == int32(0)),
            compute_perturbation_of_rho_and_theta_numpy(
                rho=rho_nnow,
                rho_ref_mc=rho_ref_mc,
                theta_v=theta_v_nnow,
                theta_ref_mc=theta_ref_mc,
            ),
            (z_rth_pr_1, z_rth_pr_2[:, :n_lev]),
        )

        (rho_ic, z_rth_pr_1, z_rth_pr_2[:, :n_lev]) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx[:n_lev] >= int32(1)),
            compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy(
                wgtfac_c=wgtfac_c[:, :n_lev],
                rho=rho_nnow,
                rho_ref_mc=rho_ref_mc,
                theta_v=theta_v_nnow,
                theta_ref_mc=theta_ref_mc,
            ),
            (rho_ic, z_rth_pr_1, z_rth_pr_2[:, :n_lev]),
        )

        (z_theta_v_pr_ic[:, :n_lev], theta_v_ic[:, :n_lev], z_th_ddz_exner_c) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx[:n_lev] >= int32(1)),
            compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
                connectivities=connectivities,
                wgtfac_c=wgtfac_c[:, :n_lev],
                z_rth_pr_2=z_rth_pr_2[:, :n_lev],
                theta_v=theta_v_nnow,
                vwind_expl_wgt=vwind_expl_wgt,
                exner_pr=exner_pr,
                d_exner_dz_ref_ic=d_exner_dz_ref_ic,
                ddqz_z_half=ddqz_z_half,
            ),
            (z_theta_v_pr_ic[:, :n_lev], theta_v_ic[:, :n_lev], z_th_ddz_exner_c),
        )

        (z_theta_v_pr_ic, theta_v_ic) = np.where(
            (vert_idx == n_lev)
            & (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo),
            set_theta_v_prime_ic_at_lower_boundary_numpy(
                wgtfacq_c=wgtfacq_c,
                z_rth_pr=z_rth_pr_2,
                theta_ref_ic=theta_ref_ic,
                z_theta_v_pr_ic=np.zeros_like(z_theta_v_pr_ic),
                theta_v_ic=np.zeros_like(theta_v_ic),
            ),
            (z_theta_v_pr_ic, theta_v_ic),
        )

        if igradp_method == 3:
            z_dexner_dz_c_2 = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (nflat_gradp <= vert_idx[:n_lev]),
                compute_approx_of_2nd_vertical_derivative_of_exner_numpy(
                    z_theta_v_pr_ic=z_theta_v_pr_ic,
                    d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
                    d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
                    z_rth_pr_2=z_rth_pr_2[:, :n_lev],
                ),
                z_dexner_dz_c_2,
            )

        (z_rth_pr_1, z_rth_pr_2[:, :n_lev]) = np.where(
            (start_cell_halo_level_2 <= horz_idx) & (horz_idx < end_cell_halo_level_2),
            compute_perturbation_of_rho_and_theta_numpy(
                rho=rho_nnow,
                rho_ref_mc=rho_ref_mc,
                theta_v=theta_v_nnow,
                theta_ref_mc=theta_ref_mc,
            ),
            (z_rth_pr_1, z_rth_pr_2[:, :n_lev]),
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
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        theta_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        wgtfacq_c = data_alloc.random_field(grid, CellDim, KDim, extend={KDim: 1})
        z_rth_pr_1 = data_alloc.zero_field(grid, CellDim, KDim)
        z_rth_pr_2 = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        z_theta_v_pr_ic = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        theta_ref_ic = data_alloc.random_field(grid, CellDim, KDim, extend={KDim: 1})
        d2dexdz2_fac1_mc = data_alloc.random_field(grid, CellDim, KDim)
        d2dexdz2_fac2_mc = data_alloc.random_field(grid, CellDim, KDim)
        wgtfac_c = data_alloc.random_field(grid, CellDim, KDim, extend={KDim: 1})
        vwind_expl_wgt = data_alloc.random_field(grid, CellDim)
        exner_pr = data_alloc.zero_field(grid, CellDim, KDim)
        d_exner_dz_ref_ic = data_alloc.random_field(grid, CellDim, KDim)
        ddqz_z_half = data_alloc.random_field(grid, CellDim, KDim)
        z_th_ddz_exner_c = data_alloc.zero_field(grid, CellDim, KDim)
        rho_ic = data_alloc.zero_field(grid, CellDim, KDim)
        z_exner_ic = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        exner_exfac = data_alloc.random_field(grid, CellDim, KDim)
        exner_nnow = data_alloc.random_field(grid, CellDim, KDim)
        exner_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        z_exner_ex_pr = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        z_dexner_dz_c_1 = data_alloc.zero_field(grid, CellDim, KDim)
        z_dexner_dz_c_2 = data_alloc.zero_field(grid, CellDim, KDim)
        theta_v_ic = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        inv_ddqz_z_full = data_alloc.random_field(grid, CellDim, KDim)
        rho_nnow = data_alloc.random_field(grid, CellDim, KDim)
        theta_v_nnow = data_alloc.random_field(grid, CellDim, KDim)

        vert_idx = data_alloc.index_field(grid, KDim, extend={KDim: 1})
        horz_idx = data_alloc.index_field(grid, CellDim)

        igradp_method = 3
        limited_area = True

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
            wgtfacq_c=wgtfacq_c,
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
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=n_lev + 1,
        )


class TestFusedMoSolveNonHydroStencil1To13Corrector(helpers.StencilTest):
    PROGRAM = fused_mo_solve_nonhydro_stencils_1_to_13_corrector
    OUTPUTS = (
        "rho_ic",
        "z_theta_v_pr_ic",
        "theta_v_ic",
        "z_th_ddz_exner_c",
    )

    # flake8: noqa: C901
    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        w: np.ndarray,
        w_concorr_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        rho_nnow: np.ndarray,
        rho_nvar: np.ndarray,
        theta_v_nnow: np.ndarray,
        theta_v_nvar: np.ndarray,
        wgtfac_c: np.ndarray,
        theta_ref_mc: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        exner_pr: np.ndarray,
        d_exner_dz_ref_ic: np.ndarray,
        rho_ic: np.ndarray,
        z_theta_v_pr_ic: np.ndarray,
        theta_v_ic: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        dtime: ta.wpfloat,
        wgt_nnow_rth: ta.wpfloat,
        wgt_nnew_rth: ta.wpfloat,
        horz_idx: np.ndarray,
        vert_idx: np.ndarray,
        start_cell_lateral_boundary_level_3: gtx.int32,
        end_cell_local: gtx.int32,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
    ) -> dict:
        lb = start_cell_lateral_boundary_level_3
        n_lev = vert_idx.shape[0] - 1
        (
            rho_ic_full,
            z_theta_v_pr_ic_full,
            theta_v_ic_full,
            z_th_ddz_exner_c_full,
        ) = compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy(
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
        )
        rho_ic[lb:end_cell_local, :n_lev] = rho_ic_full[lb:end_cell_local, :n_lev]
        z_theta_v_pr_ic[lb:end_cell_local, :n_lev] = z_theta_v_pr_ic_full[lb:end_cell_local, :n_lev]
        theta_v_ic[lb:end_cell_local, :n_lev] = theta_v_ic_full[lb:end_cell_local, :n_lev]
        z_th_ddz_exner_c[lb:end_cell_local, :n_lev] = z_th_ddz_exner_c_full[
            lb:end_cell_local, :n_lev
        ]
        return dict(
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w = data_alloc.random_field(grid, CellDim, KDim)
        w_concorr_c = data_alloc.random_field(grid, CellDim, KDim)
        ddqz_z_half = data_alloc.random_field(grid, CellDim, KDim)
        rho_nnow = data_alloc.random_field(grid, CellDim, KDim)
        rho_nvar = data_alloc.random_field(grid, CellDim, KDim)
        theta_v_nnow = data_alloc.random_field(grid, CellDim, KDim)
        theta_v_nvar = data_alloc.random_field(grid, CellDim, KDim)
        wgtfac_c = data_alloc.random_field(grid, CellDim, KDim)
        theta_ref_mc = data_alloc.random_field(grid, CellDim, KDim)
        vwind_expl_wgt = data_alloc.random_field(grid, CellDim)
        exner_pr = data_alloc.zero_field(grid, CellDim, KDim)
        d_exner_dz_ref_ic = data_alloc.random_field(grid, CellDim, KDim)
        rho_ic = data_alloc.zero_field(grid, CellDim, KDim)
        z_theta_v_pr_ic = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        theta_v_ic = data_alloc.zero_field(grid, CellDim, KDim, extend={KDim: 1})
        z_th_ddz_exner_c = data_alloc.zero_field(grid, CellDim, KDim)

        dtime = 0.9
        wgt_nnow_rth = 0.25
        wgt_nnew_rth = 0.75
        horz_idx = data_alloc.index_field(grid, CellDim)
        vert_idx = data_alloc.index_field(grid, KDim, extend={KDim: 1})

        cell_domain = h_grid.domain(CellDim)
        start_cell_lateral_boundary_level_3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        return dict(
            w=w,
            w_concorr_c=w_concorr_c,
            ddqz_z_half=ddqz_z_half,
            rho_nnow=rho_nnow,
            rho_nvar=rho_nvar,
            theta_v_nnow=theta_v_nnow,
            theta_v_nvar=theta_v_nvar,
            wgtfac_c=wgtfac_c,
            theta_ref_mc=theta_ref_mc,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
            end_cell_local=end_cell_local,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=1,
            vertical_end=grid.num_levels,
        )
