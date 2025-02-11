# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from icon4py.model.atmosphere.dycore import dycore_states, velocity_advection as advection
from icon4py.model.atmosphere.dycore.stencils import (
    fused_velocity_advection_stencil_1_to_7,
    fused_velocity_advection_stencil_8_to_13,
    fused_velocity_advection_stencil_15_to_18,
    fused_velocity_advection_stencil_19_to_20,
)
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.grid import (
    horizontal as h_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers

from . import utils


def create_vertical_params(vertical_config, grid_savepoint):
    return v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )


@pytest.mark.datatest
def test_scalfactors(savepoint_velocity_init, icon_grid, backend):
    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=None,
        interpolation_state=None,
        vertical_params=None,
        edge_params=None,
        owner_mask=None,
        backend=backend,
    )
    (cfl_w_limit, scalfac_exdiff) = velocity_advection._scale_factors_by_dtime(dtime)
    assert cfl_w_limit == savepoint_velocity_init.cfl_w_limit()
    assert scalfac_exdiff == savepoint_velocity_init.scalfac_exdiff()


@pytest.mark.datatest
def test_velocity_init(
    savepoint_velocity_init,
    interpolation_savepoint,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    backend,
):
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    assert helpers.dallclose(velocity_advection.cfl_clipping.asnumpy(), 0.0)
    assert helpers.dallclose(velocity_advection.levmask.asnumpy(), False)
    assert helpers.dallclose(velocity_advection.vcfl_dsl.asnumpy(), 0.0)

    assert velocity_advection.cfl_w_limit == 0.65
    assert velocity_advection.scalfac_exdiff == 0.05


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init",
    [
        ("mch_ch_r04b09_dsl", "2021-06-20T12:00:10.000"),
        ("exclaim_ape_R02B04", "2000-01-01T00:00:02.000"),
    ],
)
def test_verify_velocity_init_against_regular_savepoint(
    savepoint_velocity_init,
    interpolation_savepoint,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    experiment,
    backend,
):
    savepoint = savepoint_velocity_init
    dtime = savepoint.get_metadata("dtime").get("dtime")

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    assert savepoint.cfl_w_limit() == velocity_advection.cfl_w_limit / dtime
    assert savepoint.scalfac_exdiff() == velocity_advection.scalfac_exdiff / (
        dtime * (0.85 - savepoint.cfl_w_limit() * dtime)
    )


@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(1, 1)])
@pytest.mark.parametrize(
    "experiment,step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_predictor_step(
    experiment,
    istep_init,
    istep_exit,
    step_date_init,
    step_date_exit,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
    backend,
):
    sp_v = savepoint_velocity_init
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_pc=common_utils.PredictorCorrectorPair(
            sp_v.ddt_vn_apc_pc(1), sp_v.ddt_vn_apc_pc(2)
        ),
        ddt_w_adv_pc=common_utils.PredictorCorrectorPair(
            sp_v.ddt_w_adv_pc(1), sp_v.ddt_w_adv_pc(2)
        ),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
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

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    velocity_advection.run_predictor_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_w_concorr_me=sp_v.z_w_concorr_me(),
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(ntnd).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(ntnd).asnumpy()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie().asnumpy()
    icon_result_vt = savepoint_velocity_exit.vt().asnumpy()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c().asnumpy()
    icon_result_z_w_concorr_mc = savepoint_velocity_exit.z_w_concorr_mc().asnumpy()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w().asnumpy()

    # stencil 01
    assert helpers.dallclose(diagnostic_state.vt.asnumpy(), icon_result_vt, atol=1.0e-14)
    # stencil 02,05
    assert helpers.dallclose(diagnostic_state.vn_ie.asnumpy(), icon_result_vn_ie, atol=1.0e-14)

    start_edge_lateral_boundary_6 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    # stencil 07
    if not vn_only:
        assert helpers.dallclose(
            icon_result_z_v_grad_w[start_edge_lateral_boundary_6:, :],
            velocity_advection.z_v_grad_w.asnumpy()[start_edge_lateral_boundary_6:, :],
            atol=1.0e-16,
        )

    # stencil 08
    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert helpers.dallclose(
        savepoint_velocity_exit.z_ekinh().asnumpy()[start_cell_nudging:, :],
        velocity_advection.z_ekinh.asnumpy()[start_cell_nudging:, :],
    )
    # stencil 09
    assert helpers.dallclose(
        velocity_advection.z_w_concorr_mc.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev : icon_grid.num_levels
        ],
        icon_result_z_w_concorr_mc[
            start_cell_nudging:, vertical_params.nflatlev : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state.w_concorr_c.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        icon_result_w_concorr_c[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )
    # stencil 11,12,13,14
    assert helpers.dallclose(
        velocity_advection.z_w_con_c.asnumpy()[start_cell_nudging:, :],
        savepoint_velocity_exit.z_w_con_c().asnumpy()[start_cell_nudging:, :],
        atol=1.0e-15,
    )
    # stencil 16
    assert helpers.dallclose(
        diagnostic_state.ddt_w_adv_pc.predictor.asnumpy()[start_cell_nudging:, :],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
        rtol=1.0e-10,
    )
    # stencil 19
    assert helpers.dallclose(
        diagnostic_state.ddt_vn_apc_pc.predictor.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_corrector_step(
    istep_init,
    istep_exit,
    step_date_init,
    step_date_exit,
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
    sp_v = savepoint_velocity_init
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")

    assert not vn_only

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_pc=common_utils.PredictorCorrectorPair(
            sp_v.ddt_vn_apc_pc(1), sp_v.ddt_vn_apc_pc(2)
        ),
        ddt_w_adv_pc=common_utils.PredictorCorrectorPair(
            sp_v.ddt_w_adv_pc(1), sp_v.ddt_w_adv_pc(2)
        ),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
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

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    velocity_advection.run_corrector_step(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(ntnd).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(ntnd).asnumpy()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w().asnumpy()

    # stencil 07
    start_cell_lateral_boundary_level_7 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    assert helpers.dallclose(
        velocity_advection.z_v_grad_w.asnumpy()[start_cell_lateral_boundary_level_7:, :],
        icon_result_z_v_grad_w[start_cell_lateral_boundary_level_7:, :],
        atol=1e-16,
    )
    # stencil 08
    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert helpers.dallclose(
        velocity_advection.z_ekinh.asnumpy()[start_cell_nudging:, :],
        savepoint_velocity_exit.z_ekinh().asnumpy()[start_cell_nudging:, :],
    )

    # stencil 11,12,13,14
    assert helpers.dallclose(
        velocity_advection.z_w_con_c.asnumpy()[start_cell_nudging:, :],
        savepoint_velocity_exit.z_w_con_c().asnumpy()[start_cell_nudging:, :],
    )
    # stencil 16
    assert helpers.dallclose(
        diagnostic_state.ddt_w_adv_pc.corrector.asnumpy()[start_cell_nudging:, :],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
    )
    # stencil 19
    assert helpers.dallclose(
        diagnostic_state.ddt_vn_apc_pc.corrector.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=5.0e-16,
    )


@pytest.mark.dataset
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        # (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep", [(1, 2)])
def test_velocity_fussed_1_7(
    icon_grid,
    grid_savepoint,
    savepoint_velocity_1_7_init,
    savepoint_velocity_1_7_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_init,
    step_date_init,
    step_date_exit,
    istep,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)

    vn = savepoint_velocity_1_7_init.vn()
    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    z_w_concorr_me = savepoint_velocity_1_7_init.z_w_concorr_me()
    wgtfacq_e = metrics_savepoint.wgtfacq_e()
    nflatlev = grid_savepoint.nflatlev()
    c_intp = interpolation_savepoint.c_intp()
    w = savepoint_velocity_1_7_init.w()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_primal_edge_length = grid_savepoint.inv_primal_edge_length()
    tangent_orientation = grid_savepoint.tangent_orientation()
    z_vt_ie = savepoint_velocity_1_7_init.z_vt_ie()
    vt = savepoint_velocity_1_7_init.vt()
    vn_ie = savepoint_velocity_1_7_init.vn_ie()
    z_kin_hor_e = savepoint_velocity_1_7_init.z_kin_hor_e()
    z_v_grad_w = savepoint_velocity_1_7_init.z_v_grad_w()
    k = data_alloc.allocate_indices(dim=dims.KDim, grid=icon_grid, is_halfdim=True, backend=backend)

    lvn_only = False
    edge = data_alloc.allocate_indices(dim=dims.EdgeDim, grid=icon_grid, backend=backend)
    lateral_boundary_7 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
    halo_1 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO))

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
    horizontal_end = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

    vt_ref = savepoint_velocity_1_7_exit.vt()
    vn_ie_ref = savepoint_velocity_1_7_exit.vn_ie()
    z_kin_hor_e_ref = savepoint_velocity_1_7_exit.z_kin_hor_e()
    z_w_concorr_me_ref = savepoint_velocity_1_7_exit.z_w_concorr_me()
    z_v_grad_w_ref = savepoint_velocity_1_7_exit.z_v_grad_w()

    fused_velocity_advection_stencil_1_to_7.fused_velocity_advection_stencil_1_to_7.with_backend(
        backend
    )(
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        z_w_concorr_me=z_w_concorr_me,
        wgtfacq_e=wgtfacq_e,
        nflatlev=nflatlev,
        c_intp=c_intp,
        w=w,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        z_vt_ie=z_vt_ie,
        vt=vt,
        vn_ie=vn_ie,
        z_kin_hor_e=z_kin_hor_e,
        z_v_grad_w=z_v_grad_w,
        k=k,
        istep=istep,
        nlev=icon_grid.num_levels,
        lvn_only=lvn_only,
        edge=edge,
        lateral_boundary_7=lateral_boundary_7,
        halo_1=halo_1,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        verical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2V": icon_grid.get_offset_provider("E2V"),
            "V2C": icon_grid.get_offset_provider("V2C"),
            "E2C2E": icon_grid.get_offset_provider("E2C2E"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(vt_ref.asnumpy(), vt.asnumpy())
    assert helpers.dallclose(vn_ie_ref.asnumpy(), vn_ie.asnumpy())
    assert helpers.dallclose(z_kin_hor_e_ref.asnumpy(), z_kin_hor_e.asnumpy())
    assert helpers.dallclose(z_w_concorr_me_ref.asnumpy(), z_w_concorr_me.asnumpy())
    assert helpers.dallclose(z_v_grad_w_ref.asnumpy(), z_v_grad_w.asnumpy())


# TODO: can discuss the definition of k and edge


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        # (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_fused_8_13(
    icon_grid,
    grid_savepoint,
    savepoint_velocity_8_13_init,
    savepoint_velocity_8_13_exit,
    metrics_savepoint,
    interpolation_savepoint,
    istep,
    step_date_init,
    step_date_exit,
    backend,
):
    z_ekinh_ref = savepoint_velocity_8_13_exit.z_ekinh()
    w_concorr_c_ref = savepoint_velocity_8_13_exit.w_concorr_c()
    z_w_con_c_ref = savepoint_velocity_8_13_exit.z_w_con_c()

    z_kin_hor_e = savepoint_velocity_8_13_init.z_kin_hor_e()
    z_w_concorr_me = savepoint_velocity_8_13_init.z_w_concorr_m()
    w = savepoint_velocity_8_13_init.w()
    z_w_concorr_mc = savepoint_velocity_8_13_init.z_w_concorr_mc()
    w_concorr_c = savepoint_velocity_8_13_init.w_concorr_c()
    z_ekinh = savepoint_velocity_8_13_init.z_ekinh()
    z_w_con_c = savepoint_velocity_8_13_init.z_w_con_c()

    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    k = data_alloc.allocate_indices(dim=dims.KDim, grid=icon_grid, backend=backend)
    nflatlev = grid_savepoint.nflatlev()

    # vertical_lower=1; vertical_upper=nlevp1; horizontal_lower=i_startidx; horizontal_upper=i_endidx)
    fused_velocity_advection_stencil_8_to_13.fused_velocity_advection_stencil_8_to_13.with_backend(
        backend
    )(
        z_kin_hor_e=z_kin_hor_e,
        e_bln_c_s=e_bln_c_s,
        z_w_concorr_me=z_w_concorr_me,
        wgtfac_c=wgtfac_c,
        w=w,
        z_w_concorr_mc=z_w_concorr_mc,
        w_concorr_c=w_concorr_c,
        z_ekinh=z_ekinh,
        z_w_con_c=z_w_con_c,
        k=k,
        istep=istep,
        nlev=icon_grid.num_levels,
        nflatlev=nflatlev,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E"), "Koff": dims.KDim},
    )
    assert helpers.dallclose(z_ekinh_ref.asnumpy(), z_ekinh.asnumpy())
    assert helpers.dallclose(w_concorr_c_ref.asnumpy(), w_concorr_c.asnumpy())
    assert helpers.dallclose(z_w_con_c_ref.asnumpy(), z_w_con_c.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        # (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_fused_15_18(
    icon_grid,
    grid_savepoint,
    savepoint_velocity_15_18_init,
    savepoint_velocity_15_18_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_nonhydro_exit,
    backend,
    savepoint_velocity_init,
    step_date_init,
    step_date_exit,
):
    z_w_con_c = savepoint_velocity_15_18_init.z_w_con_c()
    w = savepoint_velocity_15_18_init.w()
    ddt_w_adv = savepoint_velocity_15_18_init.ddt_w_adv()
    z_v_grad_w = savepoint_velocity_15_18_init.z_v_grad_w()
    levelmask = savepoint_velocity_15_18_init.levelmask()
    z_w_con_c_full = savepoint_velocity_15_18_init.z_w_con_c_full()

    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    cfl_clipping = savepoint_nonhydro_exit.cfl_clipping()
    owner_mask = grid_savepoint.c_owner_mask()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    area = grid_savepoint.cell_areas()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    z_w_con_c_full_ref = savepoint_velocity_15_18_exit.z_w_con_c_full()
    ddt_w_adv_ref = savepoint_velocity_15_18_exit.ddt_w_adv()

    k = data_alloc.allocate_indices(dim=dims.KDim, grid=icon_grid, backend=backend)
    cell = data_alloc.allocate_indices(dim=dims.CellDim, grid=icon_grid, backend=backend)

    nrdmax = grid_savepoint.nrdmax()
    extra_diffu = True

    cell_domain = h_grid.domain(dims.EdgeDim)
    cell_lower_bound = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    cell_upper_bound = icon_grid.start_index(cell_domain(h_grid.Zone.LOCAL))

    lvn_only = False

    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()
    dtime = 2.0

    fused_velocity_advection_stencil_15_to_18.fused_velocity_advection_stencil_15_to_18.with_backend(
        backend
    )(
        z_w_con_c=z_w_con_c,
        w=w,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        ddt_w_adv=ddt_w_adv,
        e_bln_c_s=e_bln_c_s,
        z_v_grad_w=z_v_grad_w,
        levelmask=levelmask,
        cfl_clipping=cfl_clipping,
        owner_mask=owner_mask,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        z_w_con_c_full=z_w_con_c_full,
        cell=cell,
        k=k,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        cell_lower_bound=cell_lower_bound,
        cell_upper_bound=cell_upper_bound,
        nlev=icon_grid.num_levels,
        nrdmax=nrdmax,
        lvn_only=lvn_only,
        extra_diffu=extra_diffu,
        offset_provider={
            "C2E": icon_grid.get_offset_provider("C2E"),
            "C2CE": icon_grid.get_offset_provider("C2CE"),
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(z_w_con_c_full_ref.asnumpy(), z_w_con_c_full.asnumpy())
    assert helpers.dallclose(ddt_w_adv_ref.asnumpy(), ddt_w_adv.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        # (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_fused_19_20(
    icon_grid,
    grid_savepoint,
    savepoint_velocity_19_20_init,
    savepoint_velocity_19_20_exit,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    savepoint_velocity_init,
    step_date_init,
    step_date_exit,
):
    vn = savepoint_velocity_19_20_init.vn()
    z_kin_hor_e = savepoint_velocity_19_20_init.z_kin_hor_e()
    z_ekinh = savepoint_velocity_19_20_init.z_ekinh()
    vt = savepoint_velocity_19_20_init.vt()
    z_w_con_c_full = savepoint_velocity_19_20_init.z_w_con_c_full()
    vn_ie = savepoint_velocity_19_20_init.vn_ie()
    levelmask = savepoint_velocity_19_20_init.levelmask()
    ddt_vn_apc = savepoint_velocity_19_20_init.ddt_vn_apc()

    geofac_rot = interpolation_savepoint.geofac_rot()
    coeff_gradekin = metrics_savepoint.coeff_gradekin()
    f_e = grid_savepoint.f_e()
    c_lin_e = interpolation_savepoint.c_lin_e()
    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    area_edge = grid_savepoint.edge_areas()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inv_primal_edge_length()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()
    k = data_alloc.allocate_indices(dim=dims.KDim, grid=icon_grid, backend=backend)

    d_time = 2.0
    extra_diffu = True
    nrdmax = grid_savepoint.nrdmax()

    ddt_vn_apc_ref = savepoint_velocity_19_20_exit.ddt_vn_apc()

    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()

    fused_velocity_advection_stencil_19_to_20.fused_velocity_advection_stencil_19_to_20.with_backend(
        backend
    )(
        vn=vn,
        geofac_rot=geofac_rot,
        z_kin_hor_e=z_kin_hor_e,
        coeff_gradekin=coeff_gradekin,
        z_ekinh=z_ekinh,
        vt=vt,
        f_e=f_e,
        c_lin_e=c_lin_e,
        z_w_con_c_full=z_w_con_c_full,
        vn_ie=vn_ie,
        ddqz_z_full_e=ddqz_z_full_e,
        levelmask=levelmask,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        geofac_grdiv=geofac_grdiv,
        ddt_vn_apc=ddt_vn_apc,
        k=k,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        d_time=d_time,
        extra_diffu=extra_diffu,
        nlev=icon_grid.num_levels,
        nrdmax=nrdmax,
        horizontal_start=0,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "V2E": icon_grid.get_offset_provider("V2E"),
            "E2EC": icon_grid.get_offset_provider("E2EC"),
            "E2V": icon_grid.get_offset_provider("E2V"),
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2C2EO": icon_grid.get_offset_provider("E2C2EO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(ddt_vn_apc_ref.asnumpy(), ddt_vn_apc.asnumpy())
