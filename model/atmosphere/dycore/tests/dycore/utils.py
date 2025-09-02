# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional

from gt4py.next import backend as gtx_backend
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
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
        bdy_halo_c=metrics_savepoint.bdy_halo_c(),
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        time_extrapolation_parameter_for_exner=metrics_savepoint.exner_exfac(),
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
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
        pg_edgeidx_dsl=metrics_savepoint.pg_edgeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(grid_savepoint.num(dims.KDim)),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )


def construct_solve_nh_config(name: str):
    if name.lower() in "mch_ch_r04b09_dsl":
        return _mch_ch_r04b09_dsl_nonhydrostatic_config()
    elif name.lower() in "exclaim_ape_r02b04":
        return _exclaim_ape_nonhydrostatic_config()


def _mch_ch_r04b09_dsl_nonhydrostatic_config():
    """Create configuration matching the mch_chR04b09_dsl experiment."""
    config = solve_nh.NonHydrostaticConfig(
        divdamp_order=dycore_states.DivergenceDampingOrder.COMBINED,
        iau_wgt_dyn=1.0,
        fourth_order_divdamp_factor=0.004,
        max_nudging_coefficient=0.375,
    )
    return config


def _exclaim_ape_nonhydrostatic_config():
    """Create configuration for EXCLAIM APE experiment."""
    return solve_nh.NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=24,
    )


def create_vertical_params(
    vertical_config: v_grid.VerticalGridConfig,
    sp: sb.IconGridSavepoint,
):
    return v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=sp.vct_a(),
        vct_b=sp.vct_b(),
    )


def construct_diagnostics(
    init_savepoint: sb.IconNonHydroInitSavepoint,
    grid: icon_grid.IconGrid,
    backend: Optional[gtx_backend.Backend],
    swap_vertical_wind_advective_tendency: bool = False,
):
    current_index, next_index = (1, 0) if swap_vertical_wind_advective_tendency else (0, 1)
    return dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=0.0,
        theta_v_at_cells_on_half_levels=init_savepoint.theta_v_ic(),
        perturbed_exner_at_cells_on_model_levels=init_savepoint.exner_pr(),
        rho_at_cells_on_half_levels=init_savepoint.rho_ic(),
        exner_tendency_due_to_slow_physics=init_savepoint.ddt_exner_phy(),
        grf_tend_rho=init_savepoint.grf_tend_rho(),
        grf_tend_thv=init_savepoint.grf_tend_thv(),
        grf_tend_w=init_savepoint.grf_tend_w(),
        mass_flux_at_edges_on_model_levels=init_savepoint.mass_fl_e(),
        normal_wind_tendency_due_to_slow_physics_process=init_savepoint.ddt_vn_phy(),
        grf_tend_vn=init_savepoint.grf_tend_vn(),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(current_index), init_savepoint.ddt_w_adv_pc(next_index)
        ),
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
        rho_iau_increment=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=backend),
        normal_wind_iau_increment=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, backend=backend
        ),
        exner_iau_increment=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=backend),
        exner_dynamical_increment=init_savepoint.exner_dyn_incr(),
    )


def create_prognostic_states(sp) -> common_utils.TimeStepPair[prognostics.PrognosticState]:
    prognostic_state_nnow = prognostics.PrognosticState(
        w=sp.w_now(),
        vn=sp.vn_now(),
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )
    prognostic_state_nnew = prognostics.PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_nnow, prognostic_state_nnew)
    return prognostic_states
