# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import serialbox as sb


def construct_interpolation_state(
    savepoint: sb.InterpolationSavepoint,
) -> dycore_states.InterpolationState:
    grg = savepoint.geofac_grg()
    return dycore_states.InterpolationState(
        c_lin_e=savepoint.c_lin_e(),
        c_intp=savepoint.c_intp(),
        e_flx_avg=savepoint.e_flx_avg(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        geofac_rot=savepoint.geofac_rot(),
        pos_on_tplane_e_1=savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=savepoint.pos_on_tplane_e_y(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        e_bln_c_s=savepoint.e_bln_c_s(),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=savepoint.geofac_div(),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
    )


def construct_metric_state(
    metrics_savepoint: sb.MetricSavepoint, grid_savepoint: sb.IconGridSavepoint
) -> dycore_states.MetricStateNonHydro:
    return dycore_states.MetricStateNonHydro(
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        time_extrapolation_parameter_for_exner=metrics_savepoint.exner_exfac(),
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        reference_rho_at_cells_on_model_levels=metrics_savepoint.rho_ref_mc(),
        reference_theta_at_cells_on_model_levels=metrics_savepoint.theta_ref_mc(),
        exner_w_explicit_weight_parameter=metrics_savepoint.vwind_expl_wgt(),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_savepoint.d_exner_dz_ref_ic(),
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        reference_theta_at_cells_on_half_levels=metrics_savepoint.theta_ref_ic(),
        d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
        reference_rho_at_edges_on_model_levels=metrics_savepoint.rho_ref_me(),
        reference_theta_at_edges_on_model_levels=metrics_savepoint.theta_ref_me(),
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
        pg_exdist=metrics_savepoint.pg_exdist_dsl(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e(),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )


def create_vertical_params(
    vertical_config: v_grid.VerticalGridConfig,
    sp: sb.IconGridSavepoint,
) -> v_grid.VerticalGrid:
    return v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=sp.vct_a(),
        vct_b=sp.vct_b(),
    )


def construct_forcing(
    init_savepoint: sb.IconNonHydroInitSavepoint,
    grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend | None,
) -> states.DycoreForcing:
    """The tendencies and increments the dycore reads; the increments are zero (no IAU)."""

    def zeros(*field_dims: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(grid, *field_dims, allocator=backend)

    return states.DycoreForcing(
        exner_tendency_due_to_slow_physics=fw.Field(
            qty.ExnerTendencyDueToSlowPhysicsOnCellK, init_savepoint.ddt_exner_phy()
        ),
        normal_wind_tendency_due_to_slow_physics_process=fw.Field(
            qty.NormalWindTendencyDueToSlowPhysicsOnEdgeK, init_savepoint.ddt_vn_phy()
        ),
        grf_tend_rho=fw.Field(qty.GrfTendencyOfRhoOnCellK, init_savepoint.grf_tend_rho()),
        grf_tend_thv=fw.Field(qty.GrfTendencyOfThetaVOnCellK, init_savepoint.grf_tend_thv()),
        grf_tend_w=fw.Field(qty.GrfTendencyOfWOnCellKHalf, init_savepoint.grf_tend_w()),
        grf_tend_vn=fw.Field(qty.GrfTendencyOfVnOnEdgeK, init_savepoint.grf_tend_vn()),
        rho_iau_increment=fw.Field(qty.RhoIauIncrementOnCellK, zeros(dims.CellDim, dims.KDim)),
        normal_wind_iau_increment=fw.Field(
            qty.NormalWindIauIncrementOnEdgeK, zeros(dims.EdgeDim, dims.KDim)
        ),
        exner_iau_increment=fw.Field(qty.ExnerIauIncrementOnCellK, zeros(dims.CellDim, dims.KDim)),
    )


def construct_diagnostics(
    init_savepoint: sb.IconNonHydroInitSavepoint,
) -> solve_nonhydro.SolveNonhydro.Diagnostics:
    """The dycore's own scratch, from the savepoint."""
    return solve_nonhydro.SolveNonhydro.Diagnostics(
        tangential_wind=fw.Field(qty.TangentialWindOnEdgeK, init_savepoint.vt()),
        vn_on_half_levels=fw.Field(qty.VnOnEdgeKHalf, init_savepoint.vn_ie()),
        contravariant_correction_at_cells_on_half_levels=fw.Field(
            qty.ContravariantCorrectionOnCellKHalf, init_savepoint.w_concorr_c()
        ),
        theta_v_at_cells_on_half_levels=fw.Field(
            qty.ThetaVOnCellKHalf, init_savepoint.theta_v_ic()
        ),
        rho_at_cells_on_half_levels=fw.Field(qty.RhoOnCellKHalf, init_savepoint.rho_ic()),
        mass_flux_at_edges_on_model_levels=fw.Field(
            qty.MassFluxOnEdgeK, init_savepoint.mass_fl_e()
        ),
    )


def construct_dycore_diagnostics(
    init_savepoint: sb.IconNonHydroInitSavepoint,
    swap_vertical_wind_advective_tendency: bool = False,
) -> states.DycoreDiagnostics:
    """What the composer carries between the substeps, from the savepoint."""
    current_index, next_index = (1, 0) if swap_vertical_wind_advective_tendency else (0, 1)
    return states.DycoreDiagnostics(
        perturbed_exner_at_cells_on_model_levels=fw.Field(
            qty.PerturbedExnerOnCellK, init_savepoint.exner_pr()
        ),
        exner_dynamical_increment=fw.Field(
            qty.ExnerDynamicalIncrementOnCellK, init_savepoint.exner_dyn_incr()
        ),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            fw.Field(qty.NormalWindAdvectiveTendencyOnEdgeK, init_savepoint.ddt_vn_apc_pc(0)),
            fw.Field(qty.NormalWindAdvectiveTendencyOnEdgeK, init_savepoint.ddt_vn_apc_pc(1)),
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            fw.Field(
                qty.VerticalWindAdvectiveTendencyOnCellKHalf,
                init_savepoint.ddt_w_adv_pc(current_index),
            ),
            fw.Field(
                qty.VerticalWindAdvectiveTendencyOnCellKHalf,
                init_savepoint.ddt_w_adv_pc(next_index),
            ),
        ),
    )


def construct_prep_advection(
    init_savepoint: sb.IconNonHydroInitSavepoint,
    grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend | None,
) -> states.PrepAdvection:
    return states.PrepAdvection(
        vn_traj=fw.Field(qty.VnOnEdgeK, init_savepoint.vn_traj()),
        mass_flx_me=fw.Field(qty.MassFluxOnEdgeK, init_savepoint.mass_flx_me()),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=fw.Field(
            qty.MassFluxOnCellKHalf, init_savepoint.mass_flx_ic()
        ),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=fw.Field(
            qty.VolumetricFluxOnCellKHalf,
            data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim, allocator=backend),
        ),
    )


def construct_prognostic_state(
    sp: sb.IconNonHydroInitSavepoint, time_level: str
) -> states.PrognosticState:
    """The prognostics of the savepoint at the time level `now` or `new`."""

    def read(name: str) -> gtx.Field:
        return getattr(sp, f"{name}_{time_level}")()

    return states.PrognosticState(
        rho=fw.Field(qty.RhoOnCellK, read("rho")),
        w=fw.Field(qty.WOnCellKHalf, read("w")),
        vn=fw.Field(qty.VnOnEdgeK, read("vn")),
        exner=fw.Field(qty.ExnerOnCellK, read("exner")),
        theta_v=fw.Field(qty.ThetaVOnCellK, read("theta_v")),
    )


def create_prognostic_states(
    sp: sb.IconNonHydroInitSavepoint,
) -> common_utils.TimeStepPair[states.PrognosticState]:
    return common_utils.TimeStepPair(
        construct_prognostic_state(sp, "now"), construct_prognostic_state(sp, "new")
    )


def dycore_inputs(
    prognostics: states.PrognosticState, forcing: states.DycoreForcing, **scalars: Any
) -> solve_nonhydro.SolveNonhydro.Input:
    """The dycore's input view over the prognostics at `current` and the forcing."""
    return solve_nonhydro.SolveNonhydro.Input(
        rho=prognostics.rho,
        w=prognostics.w,
        vn=prognostics.vn,
        exner=prognostics.exner,
        theta_v=prognostics.theta_v,
        exner_tendency_due_to_slow_physics=forcing.exner_tendency_due_to_slow_physics,
        normal_wind_tendency_due_to_slow_physics_process=forcing.normal_wind_tendency_due_to_slow_physics_process,
        grf_tend_rho=forcing.grf_tend_rho,
        grf_tend_thv=forcing.grf_tend_thv,
        grf_tend_w=forcing.grf_tend_w,
        grf_tend_vn=forcing.grf_tend_vn,
        rho_iau_increment=forcing.rho_iau_increment,
        normal_wind_iau_increment=forcing.normal_wind_iau_increment,
        exner_iau_increment=forcing.exner_iau_increment,
        **scalars,
    )


def dycore_output(
    prognostics: states.PrognosticState,
    prep_adv: states.PrepAdvection,
    dycore_diagnostics: states.DycoreDiagnostics,
) -> solve_nonhydro.SolveNonhydro.Output:
    """The dycore's output view over the prognostics at `next`, the fluxes and the carried diagnostics."""
    return solve_nonhydro.SolveNonhydro.Output(
        rho=prognostics.rho,
        w=prognostics.w,
        vn=prognostics.vn,
        exner=prognostics.exner,
        theta_v=prognostics.theta_v,
        vn_traj=prep_adv.vn_traj,
        mass_flx_me=prep_adv.mass_flx_me,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=dycore_diagnostics.perturbed_exner_at_cells_on_model_levels,
        exner_dynamical_increment=dycore_diagnostics.exner_dynamical_increment,
        normal_wind_advective_tendency_predictor=dycore_diagnostics.normal_wind_advective_tendency.predictor,
        normal_wind_advective_tendency_corrector=dycore_diagnostics.normal_wind_advective_tendency.corrector,
        vertical_wind_advective_tendency_predictor=dycore_diagnostics.vertical_wind_advective_tendency.predictor,
        vertical_wind_advective_tendency_corrector=dycore_diagnostics.vertical_wind_advective_tendency.corrector,
    )
