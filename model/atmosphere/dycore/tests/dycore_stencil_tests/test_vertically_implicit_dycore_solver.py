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

from icon4py.model.atmosphere.dycore.dycore_states import TimeSteppingScheme
from icon4py.model.atmosphere.dycore.stencils.vertically_implicit_dycore_solver import (
    vertically_implicit_solver_at_corrector_step,
    vertically_implicit_solver_at_predictor_step,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    type_alias as ta,
)
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers

from .test_add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation_numpy,
)
from .test_apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism_numpy,
)
from .test_compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping_numpy,
)
from .test_compute_explicit_part_for_rho_and_exner import (
    compute_explicit_part_for_rho_and_exner_numpy,
)
from .test_compute_explicit_vertical_wind_from_advection_and_vertical_wind_density import (
    compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy,
)
from .test_compute_explicit_vertical_wind_speed_and_vertical_wind_times_density import (
    compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy,
)
from .test_compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables_numpy,
)
from .test_compute_solver_coefficients_matrix import (
    compute_solver_coefficients_matrix_numpy,
)
from .test_set_lower_boundary_condition_for_w_and_contravariant_correction import (
    set_lower_boundary_condition_for_w_and_contravariant_correction_numpy,
)
from .test_solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution_numpy,
)
from .test_solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep_numpy,
)
from .test_update_dynamical_exner_time_increment import update_dynamical_exner_time_increment_numpy
from .test_update_mass_volume_flux import update_mass_volume_flux_numpy


def compute_divergence_of_fluxes_of_rho_and_theta_numpy(
    connectivities,
    geofac_div: np.ndarray,
    mass_flux_at_edges_on_model_levels: np.ndarray,
    theta_v_flux_at_edges_on_model_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e = connectivities[dims.C2EDim]
    c2ce = helpers.as_1d_connectivity(c2e)
    geofac_div = np.expand_dims(geofac_div, axis=-1)

    divergence_of_mass_wp = np.sum(
        geofac_div[c2ce] * mass_flux_at_edges_on_model_levels[c2e], axis=1
    )
    divergence_of_theta_v_wp = np.sum(
        geofac_div[c2ce] * theta_v_flux_at_edges_on_model_levels[c2e], axis=1
    )
    return (divergence_of_mass_wp, divergence_of_theta_v_wp)


class TestVerticallyImplicitSolverAtPredictorStep(helpers.StencilTest):
    PROGRAM = vertically_implicit_solver_at_predictor_step
    OUTPUTS = (
        "z_contr_w_fl_l",
        "z_beta",
        "z_alpha",
        "next_w",
        "z_rho_expl",
        "z_exner_expl",
        "next_rho",
        "next_exner",
        "next_theta_v",
        "dwdz_at_cells_on_model_levels",
        "exner_dynamical_increment",
    )

    @staticmethod
    def reference(
        connectivities,
        z_contr_w_fl_l: np.ndarray,
        z_beta: np.ndarray,
        z_alpha: np.ndarray,
        next_w: np.ndarray,
        z_rho_expl: np.ndarray,
        z_exner_expl: np.ndarray,
        next_rho: np.ndarray,
        next_exner: np.ndarray,
        next_theta_v: np.ndarray,
        dwdz_at_cells_on_model_levels: np.ndarray,
        exner_dynamical_increment: np.ndarray,
        geofac_div: np.ndarray,
        mass_flux_at_edges_on_model_levels: np.ndarray,
        theta_v_flux_at_edges_on_model_levels: np.ndarray,
        predictor_vertical_wind_advective_tendency: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        rho_ic: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        current_exner: np.ndarray,
        current_rho: np.ndarray,
        current_theta_v: np.ndarray,
        current_w: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        vwind_impl_wgt: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        exner_pr: np.ndarray,
        ddt_exner_phy: np.ndarray,
        rho_iau_increment: np.ndarray,
        exner_iau_increment: np.ndarray,
        ddqz_z_half: np.ndarray,
        z_raylfac: np.ndarray,
        exner_ref_mc: np.ndarray,
        cvd_o_rd: float,
        iau_wgt_dyn: float,
        dtime: float,
        rd: float,
        cvd: float,
        cpd: float,
        rayleigh_klemp: int,
        l_vert_nested: bool,
        is_iau_active: bool,
        rayleigh_type: int,
        divdamp_type: int,
        at_first_substep: bool,
        index_of_damping_layer: int,
        n_lev: int,
        jk_start: int,
        kstart_dd3d: int,
        kstart_moist: int,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs,
    ) -> dict:
        horz_idx = np.asarray(np.arange(exner_dynamical_increment.shape[0]))
        horz_idx = horz_idx[:, np.newaxis]
        vert_idx = np.arange(exner_dynamical_increment.shape[1])
        vert_nlevp1_idx = np.arange(z_contr_w_fl_l.shape[1])

        divergence_of_mass = np.zeros_like(current_rho)
        divergence_of_theta_v = np.zeros_like(current_theta_v)
        # verified for e-9
        divergence_of_mass, divergence_of_theta_v = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
            compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                connectivities=connectivities,
                geofac_div=geofac_div,
                mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
                theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            ),
            (divergence_of_mass, divergence_of_theta_v),
        )

        z_w_expl = np.zeros_like(z_contr_w_fl_l)
        z_q = np.zeros_like(z_contr_w_fl_l)

        (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= int32(1)),
            compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
                connectivities=connectivities,
                w_nnow=current_w[:, :n_lev],
                ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency,
                z_th_ddz_exner_c=z_th_ddz_exner_c,
                rho_ic=rho_ic[:, :n_lev],
                w_concorr_c=contravariant_correction_at_cells_on_half_levels[:, :n_lev],
                vwind_expl_wgt=vwind_expl_wgt,
                dtime=dtime,
                cpd=cpd,
            ),
            (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
        )
        (z_beta, z_alpha[:, :n_lev]) = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= int32(0)),
            compute_solver_coefficients_matrix_numpy(
                connectivities=connectivities,
                exner_nnow=current_exner,
                rho_nnow=current_rho,
                theta_v_nnow=current_theta_v,
                inv_ddqz_z_full=inv_ddqz_z_full,
                vwind_impl_wgt=vwind_impl_wgt,
                theta_v_ic=theta_v_at_cells_on_half_levels[:, :n_lev],
                rho_ic=rho_ic[:, :n_lev],
                dtime=dtime,
                rd=rd,
                cvd=cvd,
            ),
            (z_beta, z_alpha[:, :n_lev]),
        )
        z_alpha = np.where(
            (horizontal_start <= horz_idx)
            & (horz_idx < horizontal_end)
            & (vert_nlevp1_idx == n_lev),
            0.0,  # _init_cell_kdim_field_with_zero_vp_numpy(connectivities=connectivities, z_alpha=z_alpha[:, :n_lev]),
            z_alpha,
        )
        z_q[:, :n_lev] = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx == int32(0)),
            0.0,
            z_q[:, :n_lev],
        )

        if not l_vert_nested:
            next_w[horizontal_start:horizontal_end, 0] = 0.0
            z_contr_w_fl_l[horizontal_start:horizontal_end, 0] = 0.0

        (next_w, z_contr_w_fl_l) = np.where(
            (horizontal_start <= horz_idx)
            & (horz_idx < horizontal_end)
            & (vert_nlevp1_idx == n_lev),
            set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
                connectivities=connectivities,
                w_concorr_c=contravariant_correction_at_cells_on_half_levels,
                z_contr_w_fl_l=z_contr_w_fl_l,
            ),
            (next_w, z_contr_w_fl_l),
        )
        # 48 and 49 are identical except for bounds
        (z_rho_expl, z_exner_expl) = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= int32(0)),
            compute_explicit_part_for_rho_and_exner_numpy(
                connectivities=connectivities,
                rho_nnow=current_rho,
                inv_ddqz_z_full=inv_ddqz_z_full,
                z_flxdiv_mass=divergence_of_mass,
                z_contr_w_fl_l=z_contr_w_fl_l,
                exner_pr=exner_pr,
                z_beta=z_beta,
                z_flxdiv_theta=divergence_of_theta_v,
                theta_v_ic=theta_v_at_cells_on_half_levels,
                ddt_exner_phy=ddt_exner_phy,
                dtime=dtime,
            ),
            (z_rho_expl, z_exner_expl),
        )

        if is_iau_active:
            (z_rho_expl, z_exner_expl) = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                add_analysis_increments_from_data_assimilation_numpy(
                    connectivities=connectivities,
                    z_rho_expl=z_rho_expl,
                    z_exner_expl=z_exner_expl,
                    rho_incr=rho_iau_increment,
                    exner_incr=exner_iau_increment,
                    iau_wgt_dyn=iau_wgt_dyn,
                ),
                (z_rho_expl, z_exner_expl),
            )
        z_q[:, :n_lev], next_w[:, :n_lev] = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= 1),
            solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
                vwind_impl_wgt=vwind_impl_wgt,
                theta_v_ic=theta_v_at_cells_on_half_levels[:, :n_lev],
                ddqz_z_half=ddqz_z_half,
                z_alpha=z_alpha,
                z_beta=z_beta,
                z_w_expl=z_w_expl,
                z_exner_expl=z_exner_expl,
                z_q_ref=z_q[:, :n_lev],
                w_ref=next_w[:, :n_lev],
                dtime=dtime,
                cpd=cpd,
            ),
            (z_q[:, :n_lev], next_w[:, :n_lev]),
        )

        next_w[:, :n_lev] = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= 1),
            solve_tridiagonal_matrix_for_w_back_substitution_numpy(
                connectivities=connectivities,
                z_q=z_q[:, :n_lev],
                w=next_w[:, :n_lev],
            ),
            next_w[:, :n_lev],
        )

        w_1 = next_w[:, 0]
        if rayleigh_type == rayleigh_klemp:
            next_w[:, :n_lev] = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= 1)
                & (vert_idx < (index_of_damping_layer + 1)),
                apply_rayleigh_damping_mechanism_numpy(
                    connectivities=connectivities,
                    z_raylfac=z_raylfac,
                    w_1=w_1,
                    w=next_w[:, :n_lev],
                ),
                next_w[:, :n_lev],
            )

        if at_first_substep:
            exner_dynamical_increment = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= kstart_moist),
                current_exner.astype(ta.vpfloat),
                exner_dynamical_increment,
            )

        next_rho, next_exner, next_theta_v = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= jk_start),
            compute_results_for_thermodynamic_variables_numpy(
                connectivities=connectivities,
                z_rho_expl=z_rho_expl,
                vwind_impl_wgt=vwind_impl_wgt,
                inv_ddqz_z_full=inv_ddqz_z_full,
                rho_ic=rho_ic,
                w=next_w,
                z_exner_expl=z_exner_expl,
                exner_ref_mc=exner_ref_mc,
                z_alpha=z_alpha,
                z_beta=z_beta,
                rho_now=current_rho,
                theta_v_now=current_theta_v,
                exner_now=current_exner,
                dtime=dtime,
                cvd_o_rd=cvd_o_rd,
            ),
            (next_rho, next_exner, next_theta_v),
        )

        # compute dw/dz for divergence damping term
        if divdamp_type >= 3:
            dwdz_at_cells_on_model_levels = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= kstart_dd3d),
                compute_dwdz_for_divergence_damping_numpy(
                    connectivities=connectivities,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    w=next_w,
                    w_concorr_c=contravariant_correction_at_cells_on_half_levels,
                ),
                dwdz_at_cells_on_model_levels,
            )

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            z_beta=z_beta,
            z_alpha=z_alpha,
            next_w=next_w,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            next_rho=next_rho,
            next_exner=next_exner,
            next_theta_v=next_theta_v,
            dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
            exner_dynamical_increment=exner_dynamical_increment,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        geofac_div = data_alloc.random_field(grid, dims.CEDim)
        mass_flux_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_flux_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        predictor_vertical_wind_advective_tendency = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        z_th_ddz_exner_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_contr_w_fl_l = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        rho_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        vwind_expl_wgt = data_alloc.random_field(grid, dims.CellDim)
        z_beta = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_alpha = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        vwind_impl_wgt = data_alloc.random_field(grid, dims.CellDim)
        theta_v_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        next_w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        z_rho_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_exner_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_pr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddt_exner_phy = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rho_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_raylfac = data_alloc.random_field(grid, dims.KDim)
        exner_ref_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        dwdz_at_cells_on_model_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_dynamical_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        l_vert_nested = False
        is_iau_active = False
        at_first_substep = True
        rayleigh_type = 2
        divdamp_type = 3
        index_of_damping_layer = 9
        jk_start = 0
        kstart_dd3d = 0
        kstart_moist = 1
        dtime = 0.9
        iau_wgt_dyn = 1.0

        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            z_beta=z_beta,
            z_alpha=z_alpha,
            next_w=next_w,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            next_rho=next_rho,
            next_exner=next_exner,
            next_theta_v=next_theta_v,
            dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
            exner_dynamical_increment=exner_dynamical_increment,
            geofac_div=geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            rho_ic=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vwind_expl_wgt=vwind_expl_wgt,
            current_exner=current_exner,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            current_w=current_w,
            inv_ddqz_z_full=inv_ddqz_z_full,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            exner_pr=exner_pr,
            ddt_exner_phy=ddt_exner_phy,
            rho_iau_increment=rho_iau_increment,
            exner_iau_increment=exner_iau_increment,
            ddqz_z_half=ddqz_z_half,
            z_raylfac=z_raylfac,
            exner_ref_mc=exner_ref_mc,
            cvd_o_rd=constants.CVD_O_RD,
            iau_wgt_dyn=iau_wgt_dyn,
            dtime=dtime,
            rd=constants.RD,
            cvd=constants.CVD,
            cpd=constants.CPD,
            rayleigh_klemp=constants.RayleighType.KLEMP.value,
            l_vert_nested=l_vert_nested,
            is_iau_active=is_iau_active,
            rayleigh_type=rayleigh_type,
            divdamp_type=divdamp_type,
            at_first_substep=at_first_substep,
            index_of_damping_layer=index_of_damping_layer,
            n_lev=grid.num_levels,
            jk_start=jk_start,
            kstart_dd3d=kstart_dd3d,
            kstart_moist=kstart_moist,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=grid.num_levels + 1,
        )


class TestVerticallyImplicitSolverAtCorrectorStep(helpers.StencilTest):
    PROGRAM = vertically_implicit_solver_at_corrector_step
    OUTPUTS = (
        "z_contr_w_fl_l",
        "z_beta",
        "z_alpha",
        "next_w",
        "z_rho_expl",
        "z_exner_expl",
        "next_rho",
        "next_exner",
        "next_theta_v",
        "mass_flx_ic",
        "vol_flx_ic",
        "exner_dynamical_increment",
    )

    @staticmethod
    def reference(
        connectivities,
        z_contr_w_fl_l: np.ndarray,
        z_beta: np.ndarray,
        z_alpha: np.ndarray,
        next_w: np.ndarray,
        z_rho_expl: np.ndarray,
        z_exner_expl: np.ndarray,
        next_rho: np.ndarray,
        next_exner: np.ndarray,
        next_theta_v: np.ndarray,
        mass_flx_ic: np.ndarray,
        vol_flx_ic: np.ndarray,
        exner_dynamical_increment: np.ndarray,
        geofac_div: np.ndarray,
        mass_flux_at_edges_on_model_levels: np.ndarray,
        theta_v_flux_at_edges_on_model_levels: np.ndarray,
        predictor_vertical_wind_advective_tendency: np.ndarray,
        corrector_vertical_wind_advective_tendency: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        rho_ic: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        current_exner: np.ndarray,
        current_rho: np.ndarray,
        current_theta_v: np.ndarray,
        current_w: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        vwind_impl_wgt: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        exner_pr: np.ndarray,
        ddt_exner_phy: np.ndarray,
        rho_iau_increment: np.ndarray,
        exner_iau_increment: np.ndarray,
        ddqz_z_half: np.ndarray,
        z_raylfac: np.ndarray,
        exner_ref_mc: np.ndarray,
        wgt_nnow_vel: float,
        wgt_nnew_vel: float,
        itime_scheme: int,
        lprep_adv: bool,
        r_nsubsteps: float,
        ndyn_substeps_var: float,
        cvd_o_rd: float,
        iau_wgt_dyn: float,
        dtime: float,
        rd: float,
        cvd: float,
        cpd: float,
        rayleigh_klemp: int,
        l_vert_nested: bool,
        is_iau_active: bool,
        rayleigh_type: int,
        at_first_substep: bool,
        at_last_substep: bool,
        index_of_damping_layer: int,
        n_lev: int,
        jk_start: int,
        kstart_moist: int,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs,
    ) -> dict:
        horz_idx = np.asarray(np.arange(exner_dynamical_increment.shape[0]))
        horz_idx = horz_idx[:, np.newaxis]
        vert_idx = np.arange(exner_dynamical_increment.shape[1])

        divergence_of_mass = np.zeros_like(current_rho)
        divergence_of_theta_v = np.zeros_like(current_theta_v)
        # verified for e-9
        divergence_of_mass, divergence_of_theta_v = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
            compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                connectivities=connectivities,
                geofac_div=geofac_div,
                mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
                theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            ),
            (divergence_of_mass, divergence_of_theta_v),
        )

        z_w_expl = np.zeros_like(z_contr_w_fl_l)
        z_q = np.zeros_like(z_contr_w_fl_l)

        if itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
            (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= int32(1))
                & (vert_idx < n_lev),
                compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy(
                    connectivities=connectivities,
                    w_nnow=current_w,
                    ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency,
                    ddt_w_adv_ntl2=corrector_vertical_wind_advective_tendency,
                    z_th_ddz_exner_c=z_th_ddz_exner_c,
                    rho_ic=rho_ic[:, :n_lev],
                    w_concorr_c=contravariant_correction_at_cells_on_half_levels[:, :n_lev],
                    vwind_expl_wgt=vwind_expl_wgt,
                    dtime=dtime,
                    wgt_nnow_vel=wgt_nnow_vel,
                    wgt_nnew_vel=wgt_nnew_vel,
                    cpd=cpd,
                ),
                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            )

            (z_beta, z_alpha[:, :n_lev]) = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= int32(0))
                & (vert_idx < n_lev),
                compute_solver_coefficients_matrix_numpy(
                    connectivities=connectivities,
                    exner_nnow=current_exner,
                    rho_nnow=current_rho,
                    theta_v_nnow=current_theta_v,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_at_cells_on_half_levels[:, :n_lev],
                    rho_ic=rho_ic[:, :n_lev],
                    dtime=dtime,
                    rd=rd,
                    cvd=cvd,
                ),
                (z_beta, z_alpha[:, :n_lev]),
            )
            z_alpha[:, :n_lev] = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx == n_lev),
                0.0,  # _init_cell_kdim_field_with_zero_vp_numpy(connectivities=connectivities, z_alpha=z_alpha[:, :n_lev]),
                z_alpha[:, :n_lev],
            )

            z_q[:, :n_lev] = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx == 0),
                0.0,
                z_q[:, :n_lev],
            )

        else:
            (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= int32(1))
                & (vert_idx < n_lev),
                compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
                    connectivities=connectivities,
                    w_nnow=current_w[:, :n_lev],
                    ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency,
                    z_th_ddz_exner_c=z_th_ddz_exner_c,
                    rho_ic=rho_ic[:, :n_lev],
                    w_concorr_c=contravariant_correction_at_cells_on_half_levels[:, :n_lev],
                    vwind_expl_wgt=vwind_expl_wgt,
                    dtime=dtime,
                    cpd=cpd,
                ),
                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            )
            (z_beta, z_alpha) = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= int32(0))
                & (vert_idx < n_lev),
                compute_solver_coefficients_matrix_numpy(
                    connectivities=connectivities,
                    exner_nnow=current_exner,
                    rho_nnow=current_rho,
                    theta_v_nnow=current_theta_v,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_at_cells_on_half_levels,
                    rho_ic=rho_ic,
                    dtime=dtime,
                    rd=rd,
                    cvd=cvd,
                ),
                (z_beta, z_alpha),
            )
            z_alpha[horizontal_start:horizontal_end, n_lev] = 0.0

            z_q[:, :n_lev] = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx == int32(0)),
                0.0,
                z_q[:, :n_lev],
            )

        if not l_vert_nested:
            next_w[horizontal_start:horizontal_end, 0] = 0.0
            z_contr_w_fl_l[horizontal_start:horizontal_end, 0] = 0.0
            # w[:, :n_lev], z_contr_w_fl_l[:, :n_lev] = np.where(
            #     (horizontal_start <= horz_idx)
            #     & (horz_idx < horizontal_end)
            #     & (vert_idx == 0),
            #     (0., 0.),
            #     # _init_two_cell_kdim_fields_with_zero_wp_numpy(
            #     #     connectivities=connectivities,
            #     #     w_nnew=w[:, :n_lev],
            #     #     z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
            #     # ),
            #     (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            # )

        (next_w[horizontal_start:horizontal_end, n_lev], z_contr_w_fl_l[horizontal_start:horizontal_end, n_lev]) = (
            set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
            connectivities,
            contravariant_correction_at_cells_on_half_levels[horizontal_start:horizontal_end, n_lev],
            z_contr_w_fl_l[horizontal_start:horizontal_end, n_lev],
        ))


        # 48 and 49 are identical except for bounds
        (z_rho_expl, z_exner_expl) = np.where(
            (horizontal_start <= horz_idx)
            & (horz_idx < horizontal_end)
            & (vert_idx >= int32(0))
            & (vert_idx < n_lev),
            compute_explicit_part_for_rho_and_exner_numpy(
                connectivities=connectivities,
                rho_nnow=current_rho,
                inv_ddqz_z_full=inv_ddqz_z_full,
                z_flxdiv_mass=divergence_of_mass,
                z_contr_w_fl_l=z_contr_w_fl_l,
                exner_pr=exner_pr,
                z_beta=z_beta,
                z_flxdiv_theta=divergence_of_theta_v,
                theta_v_ic=theta_v_at_cells_on_half_levels,
                ddt_exner_phy=ddt_exner_phy,
                dtime=dtime,
            ),
            (z_rho_expl, z_exner_expl),
        )

        if is_iau_active:
            (z_rho_expl, z_exner_expl) = np.where(
                (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
                add_analysis_increments_from_data_assimilation_numpy(
                    connectivities=connectivities,
                    z_rho_expl=z_rho_expl,
                    z_exner_expl=z_exner_expl,
                    rho_incr=rho_iau_increment,
                    exner_incr=exner_iau_increment,
                    iau_wgt_dyn=iau_wgt_dyn,
                ),
                (z_rho_expl, z_exner_expl),
            )

        z_q[:, :n_lev], next_w[:, :n_lev] = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= 1),
            solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
                vwind_impl_wgt=vwind_impl_wgt,
                theta_v_ic=theta_v_at_cells_on_half_levels[:, :n_lev],
                ddqz_z_half=ddqz_z_half,
                z_alpha=z_alpha,
                z_beta=z_beta,
                z_w_expl=z_w_expl,
                z_exner_expl=z_exner_expl,
                z_q_ref=z_q[:, :n_lev],
                w_ref=next_w[:, :n_lev],
                dtime=dtime,
                cpd=cpd,
            ),
            (z_q[:, :n_lev], next_w[:, :n_lev]),
        )

        next_w[:, :n_lev] = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= 1),
            solve_tridiagonal_matrix_for_w_back_substitution_numpy(
                connectivities=connectivities,
                z_q=z_q[:, :n_lev],
                w=next_w[:, :n_lev],
            ),
            next_w[:, :n_lev],
        )

        w_1 = next_w[:, 0]
        if rayleigh_type == rayleigh_klemp:
            next_w[:, :n_lev] = np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= 1)
                & (vert_idx < (index_of_damping_layer + 1)),
                apply_rayleigh_damping_mechanism_numpy(
                    connectivities=connectivities,
                    z_raylfac=z_raylfac,
                    w_1=w_1,
                    w=next_w[:, :n_lev],
                ),
                next_w[:, :n_lev],
            )

        next_rho, next_exner, next_theta_v = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end) & (vert_idx >= jk_start),
            compute_results_for_thermodynamic_variables_numpy(
                connectivities=connectivities,
                z_rho_expl=z_rho_expl,
                vwind_impl_wgt=vwind_impl_wgt,
                inv_ddqz_z_full=inv_ddqz_z_full,
                rho_ic=rho_ic,
                w=next_w,
                z_exner_expl=z_exner_expl,
                exner_ref_mc=exner_ref_mc,
                z_alpha=z_alpha,
                z_beta=z_beta,
                rho_now=current_rho,
                theta_v_now=current_theta_v,
                exner_now=current_exner,
                dtime=dtime,
                cvd_o_rd=cvd_o_rd,
            ),
            (next_rho, next_exner, next_theta_v),
        )

        if lprep_adv:
            if at_first_substep:
                mass_flx_ic = np.zeros_like(next_exner)
                vol_flx_ic = np.zeros_like(next_exner)

        (mass_flx_ic, vol_flx_ic) = np.where(
            (horizontal_start <= horz_idx) & (horz_idx < horizontal_end),
            update_mass_volume_flux_numpy(
                connectivities=connectivities,
                z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
                rho_ic=rho_ic[:, :n_lev],
                vwind_impl_wgt=vwind_impl_wgt,
                w=next_w[:, :n_lev],
                mass_flx_ic=mass_flx_ic,
                vol_flx_ic=vol_flx_ic,
                r_nsubsteps=r_nsubsteps,
            ),
            (mass_flx_ic, vol_flx_ic),
        )

        exner_dynamical_increment = (
            np.where(
                (horizontal_start <= horz_idx)
                & (horz_idx < horizontal_end)
                & (vert_idx >= kstart_moist)
                & (vert_idx < n_lev),
                update_dynamical_exner_time_increment_numpy(
                    connectivities=connectivities,
                    exner=next_exner,
                    ddt_exner_phy=ddt_exner_phy,
                    exner_dyn_incr=exner_dynamical_increment,
                    ndyn_substeps_var=ndyn_substeps_var,
                    dtime=dtime,
                ),
                exner_dynamical_increment,
            )
            if at_last_substep
            else exner_dynamical_increment
        )

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            z_beta=z_beta,
            z_alpha=z_alpha,
            next_w=next_w,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            next_rho=next_rho,
            next_exner=next_exner,
            next_theta_v=next_theta_v,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            exner_dynamical_increment=exner_dynamical_increment,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        geofac_div = data_alloc.random_field(grid, dims.CEDim)
        mass_flux_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_flux_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        current_w = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        predictor_vertical_wind_advective_tendency = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        corrector_vertical_wind_advective_tendency = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        z_th_ddz_exner_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_contr_w_fl_l = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        rho_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        vwind_expl_wgt = data_alloc.random_field(grid, dims.CellDim)
        z_beta = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_alpha = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        vwind_impl_wgt = data_alloc.random_field(grid, dims.CellDim)
        theta_v_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        next_w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        z_rho_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_exner_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_pr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddt_exner_phy = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rho_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_raylfac = data_alloc.random_field(grid, dims.KDim)
        exner_ref_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_dynamical_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        mass_flx_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        vol_flx_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        lprep_adv = True
        r_nsubsteps = 0.5
        rayleigh_klemp = 2
        l_vert_nested = False
        is_iau_active = False
        at_first_substep = True
        rayleigh_type = 2
        index_of_damping_layer = 9
        jk_start = 0
        at_last_substep = True
        kstart_moist = 1
        dtime = 0.9
        veladv_offctr = 0.25
        wgt_nnow_vel = 0.5 - veladv_offctr
        wgt_nnew_vel = 0.5 + veladv_offctr
        iau_wgt_dyn = 1.0
        itime_scheme = 4
        ndyn_substeps_var = 0.5

        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            z_beta=z_beta,
            z_alpha=z_alpha,
            next_w=next_w,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            next_rho=next_rho,
            next_exner=next_exner,
            next_theta_v=next_theta_v,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            exner_dynamical_increment=exner_dynamical_increment,
            geofac_div=geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            rho_ic=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vwind_expl_wgt=vwind_expl_wgt,
            current_exner=current_exner,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            current_w=current_w,
            inv_ddqz_z_full=inv_ddqz_z_full,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            exner_pr=exner_pr,
            ddt_exner_phy=ddt_exner_phy,
            rho_iau_increment=rho_iau_increment,
            exner_iau_increment=exner_iau_increment,
            ddqz_z_half=ddqz_z_half,
            z_raylfac=z_raylfac,
            exner_ref_mc=exner_ref_mc,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            itime_scheme=itime_scheme,
            lprep_adv=lprep_adv,
            r_nsubsteps=r_nsubsteps,
            ndyn_substeps_var=ndyn_substeps_var,
            cvd_o_rd=constants.CVD_O_RD,
            iau_wgt_dyn=iau_wgt_dyn,
            dtime=dtime,
            rd=constants.RD,
            cvd=constants.CVD,
            cpd=constants.CPD,
            rayleigh_klemp=rayleigh_klemp,
            l_vert_nested=l_vert_nested,
            is_iau_active=is_iau_active,
            rayleigh_type=rayleigh_type,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            index_of_damping_layer=index_of_damping_layer,
            n_lev=grid.num_levels,
            jk_start=jk_start,
            kstart_moist=kstart_moist,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=grid.num_levels + 1,
        )
