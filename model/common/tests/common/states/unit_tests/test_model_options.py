# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.model_backends import BACKENDS
import pytest

from icon4py.model.atmosphere.dycore import (
    dycore_states,
    velocity_advection as advection,
)
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.grid import (
    horizontal as h_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.testing import datatest_utils as dt_utils, test_utils
from model.atmosphere.dycore.tests.dycore import utils


def create_vertical_params(vertical_config, grid_savepoint):
    return v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:20.000", "2021-06-20T12:00:20.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_custom_backend_velocity_corrector_step(
    istep_init,
    istep_exit,
    experiment,
    step_date_init,
    step_date_exit,
    *,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
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

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=0.0,
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

    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    match_backend = [
        backend_str.split("_")
        for backend_str in BACKENDS.keys()
        if set(backend_str.split("_")).issubset(set(backend.name.split("_")))
    ][0]
    backend_options = {
        "device": match_backend[1],
        "backend_kind": match_backend[0],
    }

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend_options,
    )

    velocity_advection.run_corrector_step(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
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
