# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.stencils.vertically_implicit_dycore_solver import (
    vertically_implicit_solver_at_predictor_step,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    model_options,
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
from .test_compute_contravariant_correction_of_w import (
    compute_contravariant_correction_of_w_numpy,
)
from .test_compute_contravariant_correction_of_w_for_lower_boundary import (
    compute_contravariant_correction_of_w_for_lower_boundary_numpy,
)
from .test_compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping_numpy,
)
from .test_compute_explicit_part_for_rho_and_exner import (
    compute_explicit_part_for_rho_and_exner_numpy,
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


def compute_divergence_of_fluxes_of_rho_and_theta_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
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
        "vertical_mass_flux_at_cells_on_half_levels",
        "tridiagonal_beta_coeff_at_cells_on_model_levels",
        "tridiagonal_alpha_coeff_at_cells_on_half_levels",
        "next_w",
        "rho_explicit_term",
        "exner_explicit_term",
        "next_rho",
        "next_exner",
        "next_theta_v",
        "dwdz_at_cells_on_model_levels",
        "exner_dynamical_increment",
    )
    MARKERS = (pytest.mark.uses_concat_where,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
        tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
        tridiagonal_alpha_coeff_at_cells_on_half_levels: np.ndarray,
        next_w: np.ndarray,
        rho_explicit_term: np.ndarray,
        exner_explicit_term: np.ndarray,
        next_rho: np.ndarray,
        next_exner: np.ndarray,
        next_theta_v: np.ndarray,
        dwdz_at_cells_on_model_levels: np.ndarray,
        exner_dynamical_increment: np.ndarray,
        geofac_div: np.ndarray,
        mass_flux_at_edges_on_model_levels: np.ndarray,
        theta_v_flux_at_edges_on_model_levels: np.ndarray,
        predictor_vertical_wind_advective_tendency: np.ndarray,
        pressure_buoyancy_acceleration_at_cells_on_half_levels: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        current_exner: np.ndarray,
        current_rho: np.ndarray,
        current_theta_v: np.ndarray,
        current_w: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        exner_w_implicit_weight_parameter: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        exner_tendency_due_to_slow_physics: np.ndarray,
        rho_iau_increment: np.ndarray,
        exner_iau_increment: np.ndarray,
        ddqz_z_half: np.ndarray,
        rayleigh_damping_factor: np.ndarray,
        reference_exner_at_cells_on_model_levels: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        wgtfacq_c: np.ndarray,
        iau_wgt_dyn: float,
        dtime: float,
        is_iau_active: bool,
        rayleigh_type: int,
        divdamp_type: int,
        at_first_substep: bool,
        end_index_of_damping_layer: int,
        kstart_moist: int,
        flat_level_index_plus1: int,
        **kwargs: Any,
    ) -> dict:
        start_cell_index_nudging = kwargs["start_cell_index_nudging"]
        end_cell_index_local = kwargs["end_cell_index_local"]
        start_cell_index_lateral_lvl3 = kwargs["start_cell_index_lateral_lvl3"]
        end_cell_index_halo_lvl1 = kwargs["end_cell_index_halo_lvl1"]
        n_lev = kwargs["vertical_end_index_model_surface"] - 1
        horz_idx = np.asarray(np.arange(exner_dynamical_increment.shape[0]))
        horz_idx = horz_idx[:, np.newaxis]
        vert_idx = np.arange(exner_dynamical_increment.shape[1])

        contravariant_correction_at_cells_on_half_levels[:, :-1] = np.where(
            (start_cell_index_lateral_lvl3 <= horz_idx)
            & (horz_idx < end_cell_index_halo_lvl1)
            & (vert_idx >= flat_level_index_plus1),
            compute_contravariant_correction_of_w_numpy(
                connectivities=connectivities,
                e_bln_c_s=e_bln_c_s,
                z_w_concorr_me=contravariant_correction_at_edges_on_model_levels,
                wgtfac_c=wgtfac_c,
            ),
            contravariant_correction_at_cells_on_half_levels[:, :-1],
        )
        contravariant_correction_at_cells_on_half_levels[
            start_cell_index_lateral_lvl3:end_cell_index_halo_lvl1,
            -1,
        ] = compute_contravariant_correction_of_w_for_lower_boundary_numpy(
            connectivities=connectivities,
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=contravariant_correction_at_edges_on_model_levels,
            wgtfacq_c=wgtfacq_c,
        )[
            start_cell_index_lateral_lvl3:end_cell_index_halo_lvl1,
            -1,
        ]

        divergence_of_mass = np.zeros_like(current_rho)
        divergence_of_theta_v = np.zeros_like(current_theta_v)
        divergence_of_mass, divergence_of_theta_v = np.where(
            (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
            compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                connectivities=connectivities,
                geofac_div=geofac_div,
                mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
                theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            ),
            (divergence_of_mass, divergence_of_theta_v),
        )

        w_explicit_term = np.zeros_like(current_rho)
        tridiagonal_intermediate_result = np.zeros_like(current_rho)

        (w_explicit_term, vertical_mass_flux_at_cells_on_half_levels[:, :n_lev]) = np.where(
            (start_cell_index_nudging <= horz_idx)
            & (horz_idx < end_cell_index_local)
            & (vert_idx >= int32(1)),
            compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
                connectivities=connectivities,
                w_nnow=current_w[:, :n_lev],
                ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency[:, :n_lev],
                z_th_ddz_exner_c=pressure_buoyancy_acceleration_at_cells_on_half_levels,
                rho_ic=rho_at_cells_on_half_levels[:, :n_lev],
                w_concorr_c=contravariant_correction_at_cells_on_half_levels[:, :n_lev],
                vwind_expl_wgt=exner_w_explicit_weight_parameter,
                dtime=dtime,
                cpd=constants.CPD,
            ),
            (w_explicit_term, vertical_mass_flux_at_cells_on_half_levels[:, :n_lev]),
        )
        (
            tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels[:, :n_lev],
        ) = np.where(
            (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
            compute_solver_coefficients_matrix_numpy(
                connectivities=connectivities,
                exner_nnow=current_exner,
                rho_nnow=current_rho,
                theta_v_nnow=current_theta_v,
                inv_ddqz_z_full=inv_ddqz_z_full,
                vwind_impl_wgt=exner_w_implicit_weight_parameter,
                theta_v_ic=theta_v_at_cells_on_half_levels[:, :n_lev],
                rho_ic=rho_at_cells_on_half_levels[:, :n_lev],
                dtime=dtime,
                rd=constants.RD,
                cvd=constants.CVD,
            ),
            (
                tridiagonal_beta_coeff_at_cells_on_model_levels,
                tridiagonal_alpha_coeff_at_cells_on_half_levels[:, :n_lev],
            ),
        )
        tridiagonal_alpha_coeff_at_cells_on_half_levels[
            start_cell_index_nudging:end_cell_index_local, n_lev
        ] = 0.0
        tridiagonal_intermediate_result[start_cell_index_nudging:end_cell_index_local, 0] = 0.0

        next_w[start_cell_index_nudging:end_cell_index_local, 0] = 0.0
        vertical_mass_flux_at_cells_on_half_levels[
            start_cell_index_nudging:end_cell_index_local, 0
        ] = 0.0

        (
            next_w[start_cell_index_nudging:end_cell_index_local, n_lev],
            vertical_mass_flux_at_cells_on_half_levels[
                start_cell_index_nudging:end_cell_index_local, n_lev
            ],
        ) = set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
            connectivities,
            w_concorr_c=contravariant_correction_at_cells_on_half_levels[
                start_cell_index_nudging:end_cell_index_local, n_lev
            ],
            z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels[
                start_cell_index_nudging:end_cell_index_local, n_lev
            ],
        )
        # 48 and 49 are identical except for bounds
        (rho_explicit_term, exner_explicit_term) = np.where(
            (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
            compute_explicit_part_for_rho_and_exner_numpy(
                connectivities=connectivities,
                rho_nnow=current_rho,
                inv_ddqz_z_full=inv_ddqz_z_full,
                z_flxdiv_mass=divergence_of_mass,
                z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
                exner_pr=perturbed_exner_at_cells_on_model_levels,
                z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
                z_flxdiv_theta=divergence_of_theta_v,
                theta_v_ic=theta_v_at_cells_on_half_levels,
                ddt_exner_phy=exner_tendency_due_to_slow_physics,
                dtime=dtime,
            ),
            (rho_explicit_term, exner_explicit_term),
        )

        if is_iau_active:
            (rho_explicit_term, exner_explicit_term) = np.where(
                (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
                add_analysis_increments_from_data_assimilation_numpy(
                    connectivities=connectivities,
                    z_rho_expl=rho_explicit_term,
                    z_exner_expl=exner_explicit_term,
                    rho_incr=rho_iau_increment,
                    exner_incr=exner_iau_increment,
                    iau_wgt_dyn=iau_wgt_dyn,
                ),
                (rho_explicit_term, exner_explicit_term),
            )
        (tridiagonal_intermediate_result, next_w[:, :n_lev]) = np.where(
            (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
            solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
                vwind_impl_wgt=exner_w_implicit_weight_parameter,
                theta_v_ic=theta_v_at_cells_on_half_levels[:, :n_lev],
                ddqz_z_half=ddqz_z_half,
                z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
                z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
                z_w_expl=w_explicit_term,
                z_exner_expl=exner_explicit_term,
                z_q_ref=tridiagonal_intermediate_result,
                w_ref=next_w[:, :n_lev],
                dtime=dtime,
                cpd=constants.CPD,
            ),
            (tridiagonal_intermediate_result, next_w[:, :n_lev]),
        )

        next_w[:, :n_lev] = np.where(
            (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
            solve_tridiagonal_matrix_for_w_back_substitution_numpy(
                connectivities=connectivities,
                z_q=tridiagonal_intermediate_result[:, :n_lev],
                w=next_w[:, :n_lev],
            ),
            next_w[:, :n_lev],
        )

        w_1 = next_w[:, 0]
        if rayleigh_type == model_options.RayleighType.KLEMP:
            next_w[:, :n_lev] = np.where(
                (start_cell_index_nudging <= horz_idx)
                & (horz_idx < end_cell_index_local)
                & (vert_idx >= 1)
                & (vert_idx < (end_index_of_damping_layer + 1)),
                apply_rayleigh_damping_mechanism_numpy(
                    connectivities=connectivities,
                    z_raylfac=rayleigh_damping_factor,
                    w_1=w_1,
                    w=next_w[:, :n_lev],
                ),
                next_w[:, :n_lev],
            )

        if at_first_substep:
            exner_dynamical_increment = np.where(
                (start_cell_index_nudging <= horz_idx)
                & (horz_idx < end_cell_index_local)
                & (vert_idx >= kstart_moist),
                current_exner.astype(ta.vpfloat),
                exner_dynamical_increment,
            )

        next_rho, next_exner, next_theta_v = np.where(
            (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
            compute_results_for_thermodynamic_variables_numpy(
                connectivities=connectivities,
                z_rho_expl=rho_explicit_term,
                vwind_impl_wgt=exner_w_implicit_weight_parameter,
                inv_ddqz_z_full=inv_ddqz_z_full,
                rho_ic=rho_at_cells_on_half_levels,
                w=next_w,
                z_exner_expl=exner_explicit_term,
                exner_ref_mc=reference_exner_at_cells_on_model_levels,
                z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
                z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
                rho_now=current_rho,
                theta_v_now=current_theta_v,
                exner_now=current_exner,
                dtime=dtime,
            ),
            (next_rho, next_exner, next_theta_v),
        )

        # compute dw/dz for divergence damping term
        if divdamp_type >= 3:
            dwdz_at_cells_on_model_levels = np.where(
                (start_cell_index_nudging <= horz_idx) & (horz_idx < end_cell_index_local),
                compute_dwdz_for_divergence_damping_numpy(
                    connectivities=connectivities,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    w=next_w,
                    w_concorr_c=contravariant_correction_at_cells_on_half_levels,
                ),
                dwdz_at_cells_on_model_levels,
            )

        return dict(
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w=next_w,
            rho_explicit_term=rho_explicit_term,
            exner_explicit_term=exner_explicit_term,
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
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        pressure_buoyancy_acceleration_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        vertical_mass_flux_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        rho_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, low=1.0e-5
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        exner_w_explicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim)
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5)
        current_w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5)
        exner_w_implicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim)
        theta_v_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, low=1.0e-5
        )
        perturbed_exner_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        exner_tendency_due_to_slow_physics = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rho_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5)
        rayleigh_damping_factor = data_alloc.random_field(grid, dims.KDim)
        reference_exner_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-5
        )
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim, low=1.0e-5, high=0.99999)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5, high=0.99999)
        wgtfacq_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-5, high=0.99999)

        tridiagonal_beta_coeff_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        tridiagonal_alpha_coeff_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        next_w = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        rho_explicit_term = data_alloc.constant_field(grid, 1.0e-5, dims.CellDim, dims.KDim)
        exner_explicit_term = data_alloc.constant_field(grid, 1.0e-5, dims.CellDim, dims.KDim)
        next_rho = data_alloc.constant_field(grid, 1.0e-5, dims.CellDim, dims.KDim)
        next_exner = data_alloc.constant_field(grid, 1.0e-5, dims.CellDim, dims.KDim)
        next_theta_v = data_alloc.constant_field(grid, 1.0e-5, dims.CellDim, dims.KDim)
        dwdz_at_cells_on_model_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        exner_dynamical_increment = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        is_iau_active = True
        at_first_substep = True
        rayleigh_type = 2
        divdamp_type = 3
        end_index_of_damping_layer = 3
        kstart_moist = 1
        flat_level_index_plus1 = 3
        dtime = 0.001
        iau_wgt_dyn = 1.0

        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        start_cell_index_lateral_lvl3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        end_cell_index_halo_lvl1 = grid.end_index(cell_domain(h_grid.Zone.HALO))

        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w=next_w,
            rho_explicit_term=rho_explicit_term,
            exner_explicit_term=exner_explicit_term,
            next_rho=next_rho,
            next_exner=next_exner,
            next_theta_v=next_theta_v,
            dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
            exner_dynamical_increment=exner_dynamical_increment,
            geofac_div=geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            pressure_buoyancy_acceleration_at_cells_on_half_levels=pressure_buoyancy_acceleration_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            current_exner=current_exner,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            current_w=current_w,
            inv_ddqz_z_full=inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
            rho_iau_increment=rho_iau_increment,
            exner_iau_increment=exner_iau_increment,
            ddqz_z_half=ddqz_z_half,
            rayleigh_damping_factor=rayleigh_damping_factor,
            reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            iau_wgt_dyn=iau_wgt_dyn,
            dtime=dtime,
            is_iau_active=is_iau_active,
            rayleigh_type=rayleigh_type,
            divdamp_type=divdamp_type,
            at_first_substep=at_first_substep,
            end_index_of_damping_layer=end_index_of_damping_layer,
            kstart_moist=kstart_moist,
            flat_level_index_plus1=flat_level_index_plus1,
            start_cell_index_nudging=start_cell_nudging,
            end_cell_index_local=end_cell_local,
            start_cell_index_lateral_lvl3=start_cell_index_lateral_lvl3,
            end_cell_index_halo_lvl1=end_cell_index_halo_lvl1,
            vertical_start_index_model_top=gtx.int32(0),
            vertical_end_index_model_surface=gtx.int32(grid.num_levels + 1),
        )
