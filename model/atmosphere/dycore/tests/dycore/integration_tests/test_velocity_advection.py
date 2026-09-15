# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro
from icon4py.model.atmosphere.dycore.stencils import (
    velocity_advection_corrector,
    velocity_advection_predictor,
)
from icon4py.model.common import dimension as dims, type_alias as ta, utils as common_utils
from icon4py.model.common.grid import (
    horizontal as h_grid,
    icon,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import nonhydro_states, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, serialbox, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


log = logging.getLogger(__name__)


def _compare_cfl(
    *,
    vertical_cfl: np.ndarray,
    icon_result_cfl_clipping: np.ndarray,
    icon_result_max_vcfl_dyn: float,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> None:
    cfl_clipping_mask = np.where(np.abs(vertical_cfl) > 0.0, True, False)
    assert (
        cfl_clipping_mask[horizontal_start:horizontal_end, vertical_start:vertical_end]
        == icon_result_cfl_clipping[horizontal_start:horizontal_end, vertical_start:vertical_end]
    ).all()

    assert vertical_cfl[horizontal_start:horizontal_end, :].max() == icon_result_max_vcfl_dyn


def create_vertical_params(
    vertical_config: v_grid.VerticalGridConfig, grid_savepoint: serialbox.IconGridSavepoint
) -> v_grid.VerticalGrid:
    return v_grid.VerticalGrid(
        config=vertical_config, vct_a=grid_savepoint.vct_a(), vct_b=grid_savepoint.vct_b()
    )


@pytest.mark.embedded_static_args
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init",
    [
        (test_defs.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:10.000"),
        (test_defs.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000"),
    ],
)
def test_scale_factors_by_dtime(experiment, step_date_init, savepoint_velocity_init):
    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    cfl_w_limit, scalfac_exdiff = solve_nonhydro._velocity_advection_scale_factors(dtime)
    assert cfl_w_limit == savepoint_velocity_init.cfl_w_limit()
    assert scalfac_exdiff == savepoint_velocity_init.scalfac_exdiff()


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            test_defs.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
        (test_defs.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_predictor_step(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
    backend,
    caplog,
):
    caplog.set_level(logging.WARN)
    init_savepoint = savepoint_velocity_init
    vn_only = init_savepoint.vn_only()
    dtime = init_savepoint.get_metadata("dtime").get("dtime")

    diagnostic_state = nonhydro_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, backend),
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
        theta_v_at_cells_on_half_levels=None,
        perturbed_exner_at_cells_on_model_levels=None,
        rho_at_cells_on_half_levels=None,
        exner_tendency_due_to_slow_physics=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_flux_at_edges_on_model_levels=None,
        normal_wind_tendency_due_to_slow_physics_process=None,
        grf_tend_vn=None,
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(0), init_savepoint.ddt_w_adv_pc(1)
        ),
        rho_iau_increment=None,
        normal_wind_iau_increment=None,
        exner_iau_increment=None,
        exner_dynamical_increment=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=init_savepoint.w(),
        vn=init_savepoint.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    vertical_config = experiment.config.vertical_grid
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    end_edge_halo_level_2 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))
    start_cell_lateral_boundary_level_4 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
    )
    start_edge_lateral_boundary_level_5 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )
    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    vertical_cfl = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KHalfDim, allocator=backend, dtype=ta.vpfloat
    )
    cfl_w_limit, scalfac_exdiff = solve_nonhydro._velocity_advection_scale_factors(dtime)

    contravariant_correction_at_edges_on_model_levels = init_savepoint.z_w_concorr_me()
    horizontal_kinetic_energy_at_edges_on_model_levels = init_savepoint.z_kin_hor_e()
    tangential_wind_on_half_levels = init_savepoint.z_vt_ie()

    velocity_advection_predictor.compute_velocity_advection_in_predictor_step.with_backend(backend)(
        tangential_wind=diagnostic_state.tangential_wind,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=diagnostic_state.vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        contravariant_correction_at_cells_on_half_levels=diagnostic_state.contravariant_correction_at_cells_on_half_levels,
        vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.predictor,
        vertical_cfl=vertical_cfl,
        normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.predictor,
        vn=prognostic_state.vn,
        w=prognostic_state.w,
        rbf_vec_coeff_e=interpolation_state.rbf_vec_coeff_e,
        wgtfac_e=metric_state_nonhydro.wgtfac_e,
        wgtfacq_e=metric_state_nonhydro.wgtfacq_e,
        ddxn_z_full=metric_state_nonhydro.ddxn_z_full,
        ddxt_z_full=metric_state_nonhydro.ddxt_z_full,
        coeff1_dwdz=metric_state_nonhydro.coeff1_dwdz,
        coeff2_dwdz=metric_state_nonhydro.coeff2_dwdz,
        c_intp=interpolation_state.c_intp,
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        tangent_orientation=edge_geometry.tangent_orientation,
        e_bln_c_s=interpolation_state.e_bln_c_s,
        wgtfac_c=metric_state_nonhydro.wgtfac_c,
        ddqz_z_half=metric_state_nonhydro.ddqz_z_half,
        area=cell_geometry.area,
        geofac_n2s=interpolation_state.geofac_n2s,
        owner_mask=grid_savepoint.c_owner_mask(),
        coriolis_frequency=edge_geometry.coriolis_frequency,
        geofac_rot=interpolation_state.geofac_rot,
        coeff_gradekin=metric_state_nonhydro.coeff_gradekin,
        c_lin_e=interpolation_state.c_lin_e,
        ddqz_z_full_e=metric_state_nonhydro.ddqz_z_full_e,
        area_edge=edge_geometry.edge_areas,
        geofac_grdiv=interpolation_state.geofac_grdiv,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        skip_compute_predictor_vertical_advection=vn_only,
        apply_extra_diffusion_on_vn=True,
        nflatlev=vertical_params.nflatlev,
        end_index_of_damping_layer=vertical_params.end_index_of_damping_layer,
        start_edge_lateral_boundary_level_5=start_edge_lateral_boundary_level_5,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_cell_lateral_boundary_level_4=start_cell_lateral_boundary_level_4,
        end_cell_halo=end_cell_halo,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        vertical_start=gtx.int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E": icon_grid.get_connectivity("C2E"),
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2C2E": icon_grid.get_connectivity("E2C2E"),
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
            "E2V": icon_grid.get_connectivity("E2V"),
            "V2C": icon_grid.get_connectivity("V2C"),
            "V2E": icon_grid.get_connectivity("V2E"),
        },
    )
    solve_nonhydro._update_max_vertical_cfl(
        diagnostic_state, vertical_cfl, start_cell_lateral_boundary_level_4, end_cell_halo
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(0).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(0).asnumpy()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie().asnumpy()
    icon_result_vt = savepoint_velocity_exit.vt().asnumpy()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c().asnumpy()
    icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()

    assert test_utils.dallclose(
        diagnostic_state.tangential_wind.asnumpy(), icon_result_vt, atol=1.0e-14
    )

    assert test_utils.dallclose(
        diagnostic_state.vn_on_half_levels.asnumpy(), icon_result_vn_ie, atol=1.0e-14
    )

    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert test_utils.dallclose(
        diagnostic_state.contravariant_correction_at_cells_on_half_levels.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        icon_result_w_concorr_c[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )

    assert test_utils.dallclose(
        diagnostic_state.vertical_wind_advective_tendency.predictor.asnumpy()[
            start_cell_nudging:, :
        ],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
        rtol=1.0e-10,
    )

    assert test_utils.dallclose(
        diagnostic_state.normal_wind_advective_tendency.predictor.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=1.0e-15,
    )

    # ICON sets z_vt_ie on the top half level unconditionally
    # (mo_velocity_advection.f90:300) whereas icon4py leaves the whole field untouched
    # when the predictor's vertical advection is skipped.
    first_comparable_half_level = 1 if vn_only else 0
    assert test_utils.dallclose(
        tangential_wind_on_half_levels.asnumpy()[:, first_comparable_half_level:],
        savepoint_velocity_exit.z_vt_ie().asnumpy()[:, first_comparable_half_level:],
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    assert test_utils.dallclose(
        horizontal_kinetic_energy_at_edges_on_model_levels.asnumpy(),
        savepoint_velocity_exit.z_kin_hor_e().asnumpy(),
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    assert test_utils.dallclose(
        contravariant_correction_at_edges_on_model_levels.asnumpy(),
        savepoint_velocity_exit.z_w_concorr_me().asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )

    assert diagnostic_state.max_vertical_cfl == icon_result_max_vcfl_dyn

    _compare_cfl(
        vertical_cfl=vertical_cfl.asnumpy(),
        icon_result_cfl_clipping=savepoint_velocity_exit.cfl_clipping().asnumpy(),
        icon_result_max_vcfl_dyn=icon_result_max_vcfl_dyn,
        horizontal_start=start_cell_lateral_boundary_level_4,
        horizontal_end=end_cell_halo,
        vertical_start=max(2, grid_savepoint.nrdmax() - 2),
        vertical_end=icon_grid.num_levels - 3,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            test_defs.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
        (test_defs.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_corrector_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    istep_exit,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
):
    init_savepoint = savepoint_velocity_init
    vn_only = init_savepoint.vn_only()
    dtime = init_savepoint.get_metadata("dtime").get("dtime")

    assert not vn_only

    diagnostic_state = nonhydro_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, backend),
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
        theta_v_at_cells_on_half_levels=None,
        perturbed_exner_at_cells_on_model_levels=None,
        rho_at_cells_on_half_levels=None,
        exner_tendency_due_to_slow_physics=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_flux_at_edges_on_model_levels=None,
        normal_wind_tendency_due_to_slow_physics_process=None,
        grf_tend_vn=None,
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(0), init_savepoint.ddt_w_adv_pc(1)
        ),
        rho_iau_increment=None,
        normal_wind_iau_increment=None,
        exner_iau_increment=None,  # sp.exner_incr(),
        exner_dynamical_increment=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=init_savepoint.w(),
        vn=init_savepoint.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)

    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    vertical_config = experiment.config.vertical_grid
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))
    start_cell_lateral_boundary_level_4 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
    )
    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    vertical_cfl = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KHalfDim, allocator=backend, dtype=ta.vpfloat
    )
    cfl_w_limit, scalfac_exdiff = solve_nonhydro._velocity_advection_scale_factors(dtime)

    velocity_advection_corrector.compute_velocity_advection_in_corrector_step.with_backend(backend)(
        vertical_wind_advective_tendency=diagnostic_state.vertical_wind_advective_tendency.corrector,
        vertical_cfl=vertical_cfl,
        normal_wind_advective_tendency=diagnostic_state.normal_wind_advective_tendency.corrector,
        vn=prognostic_state.vn,
        w=prognostic_state.w,
        tangential_wind=diagnostic_state.tangential_wind,
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
        vn_on_half_levels=diagnostic_state.vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        contravariant_correction_at_cells_on_half_levels=diagnostic_state.contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=metric_state_nonhydro.coeff1_dwdz,
        coeff2_dwdz=metric_state_nonhydro.coeff2_dwdz,
        c_intp=interpolation_state.c_intp,
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        tangent_orientation=edge_geometry.tangent_orientation,
        e_bln_c_s=interpolation_state.e_bln_c_s,
        ddqz_z_half=metric_state_nonhydro.ddqz_z_half,
        area=cell_geometry.area,
        geofac_n2s=interpolation_state.geofac_n2s,
        owner_mask=grid_savepoint.c_owner_mask(),
        coriolis_frequency=edge_geometry.coriolis_frequency,
        geofac_rot=interpolation_state.geofac_rot,
        coeff_gradekin=metric_state_nonhydro.coeff_gradekin,
        c_lin_e=interpolation_state.c_lin_e,
        ddqz_z_full_e=metric_state_nonhydro.ddqz_z_full_e,
        area_edge=edge_geometry.edge_areas,
        geofac_grdiv=interpolation_state.geofac_grdiv,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        apply_extra_diffusion_on_vn=True,
        end_index_of_damping_layer=vertical_params.end_index_of_damping_layer,
        start_cell_lateral_boundary_level_4=start_cell_lateral_boundary_level_4,
        end_cell_halo=end_cell_halo,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        vertical_start=gtx.int32(0),
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E": icon_grid.get_connectivity("C2E"),
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
            "E2V": icon_grid.get_connectivity("E2V"),
            "V2C": icon_grid.get_connectivity("V2C"),
            "V2E": icon_grid.get_connectivity("V2E"),
        },
    )
    solve_nonhydro._update_max_vertical_cfl(
        diagnostic_state, vertical_cfl, start_cell_lateral_boundary_level_4, end_cell_halo
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(1).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(1).asnumpy()
    icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()

    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert test_utils.dallclose(
        diagnostic_state.vertical_wind_advective_tendency.corrector.asnumpy()[
            start_cell_nudging:, :
        ],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
    )
    assert test_utils.dallclose(
        diagnostic_state.normal_wind_advective_tendency.corrector.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=5.0e-16,
    )

    assert diagnostic_state.max_vertical_cfl == icon_result_max_vcfl_dyn

    _compare_cfl(
        vertical_cfl=vertical_cfl.asnumpy(),
        icon_result_cfl_clipping=savepoint_velocity_exit.cfl_clipping().asnumpy(),
        icon_result_max_vcfl_dyn=icon_result_max_vcfl_dyn,
        horizontal_start=start_cell_lateral_boundary_level_4,
        horizontal_end=end_cell_halo,
        vertical_start=max(2, grid_savepoint.nrdmax() - 2),
        vertical_end=icon_grid.num_levels - 3,
    )
