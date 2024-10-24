# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from icon4pytools.py2fgen.wrappers.common import backend

import icon4py.model.common.dimension as dims
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states, diffusion_utils
from icon4py.model.common import settings
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry import CellParams, EdgeParams
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
    reference_funcs as ref_funcs,
    serialbox_utils as sb,
)

from .utils import (
    compare_dace_orchestration_multiple_steps,
    construct_diffusion_config,
    diff_multfac_vn_numpy,
    diffusion_instance,  # noqa
    smag_limit_numpy,
    verify_diffusion_fields,
)


def test_diffusion_coefficients_with_hdiff_efdt_ratio(experiment):
    config = construct_diffusion_config(experiment, ndyn_substeps=5)
    config.hdiff_efdt_ratio = 1.0
    config.hdiff_w_efdt_ratio = 2.0

    params = diffusion.DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K6 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(1.0 / 72.0, abs=1e-12)


def test_diffusion_coefficients_without_hdiff_efdt_ratio(experiment):
    config = construct_diffusion_config(experiment)
    config.hdiff_efdt_ratio = 0.0
    config.hdiff_w_efdt_ratio = 0.0

    params = diffusion.DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K6 == 0.0
    assert params.K4W == 0.0


def test_smagorinski_factor_for_diffusion_type_4(experiment):
    config = construct_diffusion_config(experiment, ndyn_substeps=5)
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 4

    params = diffusion.DiffusionParams(config)
    assert len(params.smagorinski_factor) == 1
    assert params.smagorinski_factor[0] == pytest.approx(0.15, abs=1e-16)
    assert params.smagorinski_height is None


def test_smagorinski_heights_diffusion_type_5_are_consistent(
    experiment,
):
    config = construct_diffusion_config(experiment, ndyn_substeps=5)
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 5

    params = diffusion.DiffusionParams(config)
    assert len(params.smagorinski_height) == 4
    assert min(params.smagorinski_height) == params.smagorinski_height[0]
    assert max(params.smagorinski_height) == params.smagorinski_height[-1]
    assert params.smagorinski_height[0] < params.smagorinski_height[1]
    assert params.smagorinski_height[1] < params.smagorinski_height[3]
    assert params.smagorinski_height[2] != params.smagorinski_height[1]
    assert params.smagorinski_height[2] != params.smagorinski_height[3]


def test_smagorinski_factor_diffusion_type_5(experiment):
    params = diffusion.DiffusionParams(construct_diffusion_config(experiment, ndyn_substeps=5))
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))


@pytest.mark.datatest
def test_diffusion_init(
    savepoint_diffusion_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    experiment,
    step_date_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
):
    config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    meta = savepoint_diffusion_init.get_metadata("linit", "date")

    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )

    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion_granule = diffusion.Diffusion(backend=backend)
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )

    assert diffusion_granule.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float
    )

    assert helpers.dallclose(diffusion_granule.v_vert.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.u_vert.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.kh_smag_ec.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.kh_smag_e.asnumpy(), 0.0)

    shape_k = (icon_grid.num_levels,)
    expected_smag_limit = smag_limit_numpy(
        diff_multfac_vn_numpy,
        shape_k,
        additional_parameters.K4,
        config.substep_as_float,
    )

    assert (
        diffusion_granule.smag_offset == 0.25 * additional_parameters.K4 * config.substep_as_float
    )
    assert helpers.dallclose(diffusion_granule.smag_limit.asnumpy(), expected_smag_limit)

    expected_diff_multfac_vn = diff_multfac_vn_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float
    )
    assert helpers.dallclose(diffusion_granule.diff_multfac_vn.asnumpy(), expected_diff_multfac_vn)
    expected_enh_smag_fac = ref_funcs.enhanced_smagorinski_factor_numpy(
        additional_parameters.smagorinski_factor,
        additional_parameters.smagorinski_height,
        grid_savepoint.vct_a().asnumpy(),
    )
    assert helpers.dallclose(diffusion_granule.enh_smag_fac.asnumpy(), expected_enh_smag_fac)


def _verify_init_values_against_savepoint(
    savepoint: sb.IconDiffusionInitSavepoint, diffusion_granule: diffusion.Diffusion
):
    dtime = savepoint.get_metadata("dtime")["dtime"]

    assert savepoint.nudgezone_diff() == diffusion_granule.nudgezone_diff
    assert savepoint.bdy_diff() == diffusion_granule.bdy_diff
    assert savepoint.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v
    assert savepoint.smag_offset() == diffusion_granule.smag_offset
    assert savepoint.diff_multfac_w() == diffusion_granule.diff_multfac_w

    # this is done in diffusion.run(...) because it depends on the dtime
    diffusion_utils.scale_k.with_backend(backend)(
        diffusion_granule.enh_smag_fac,
        dtime,
        diffusion_granule.diff_multfac_smag,
        offset_provider={},
    )
    assert helpers.dallclose(
        diffusion_granule.enh_smag_fac.asnumpy(), savepoint.enh_smag_fac(), rtol=1e-7
    )
    assert helpers.dallclose(
        diffusion_granule.diff_multfac_smag.asnumpy(), savepoint.diff_multfac_smag(), rtol=1e-7
    )

    assert helpers.dallclose(diffusion_granule.smag_limit.asnumpy(), savepoint.smag_limit())
    assert helpers.dallclose(
        diffusion_granule.diff_multfac_n2w.asnumpy(), savepoint.diff_multfac_n2w()
    )
    assert helpers.dallclose(
        diffusion_granule.diff_multfac_vn.asnumpy(), savepoint.diff_multfac_vn()
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment,step_date_init",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000"),
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:20.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:04.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_verify_diffusion_init_against_savepoint(
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_diffusion_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
):
    config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion_granule = diffusion.Diffusion(backend=backend)
    diffusion_granule.init(
        icon_grid,
        config,
        additional_parameters,
        vertical_params,
        metric_state,
        interpolation_state,
        edge_params,
        cell_params,
    )

    _verify_init_values_against_savepoint(savepoint_diffusion_init, diffusion_granule)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_run_diffusion_single_step(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
    diffusion_instance,  # noqa: F811
):
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )

    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state = savepoint_diffusion_init.construct_prognostics()

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )

    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    config = construct_diffusion_config(experiment, ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion_instance  # the fixture makes sure that the orchestrator cache is cleared properly between pytest runs -if applicable-
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_init)
    assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    diffusion_granule.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )

    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_exit)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_run_diffusion_multiple_steps(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
    diffusion_instance,  # noqa: F811
):
    if settings.dace_orchestration is None:
        raise pytest.skip("This test is only executed for `--dace-orchestration=True`.")

    ######################################################################
    # Diffusion initialization
    ######################################################################
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )

    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    config = construct_diffusion_config(experiment, ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    ######################################################################
    # DaCe NON-Orchestrated Backend
    ######################################################################
    settings.dace_orchestration = None

    diagnostic_state_dace_non_orch = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state_dace_non_orch = savepoint_diffusion_init.construct_prognostics()

    diffusion_granule = diffusion.Diffusion(backend=backend)
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    for _ in range(3):
        diffusion_granule.run(
            diagnostic_state=diagnostic_state_dace_non_orch,
            prognostic_state=prognostic_state_dace_non_orch,
            dtime=dtime,
        )

    ######################################################################
    # DaCe Orchestrated Backend
    ######################################################################
    settings.dace_orchestration = True

    diagnostic_state_dace_orch = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state_dace_orch = savepoint_diffusion_init.construct_prognostics()

    diffusion_granule = diffusion_instance  # the fixture makes sure that the orchestrator cache is cleared properly between pytest runs -if applicable-
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    for _ in range(3):
        diffusion_granule.run(
            diagnostic_state=diagnostic_state_dace_orch,
            prognostic_state=prognostic_state_dace_orch,
            dtime=dtime,
        )

    ######################################################################
    # Verify the results
    ######################################################################
    compare_dace_orchestration_multiple_steps(
        diagnostic_state_dace_non_orch, diagnostic_state_dace_orch
    )
    compare_dace_orchestration_multiple_steps(
        prognostic_state_dace_non_orch, prognostic_state_dace_orch
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize("linit", [True])
def test_run_diffusion_initial_step(
    experiment,
    linit,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    backend,
    diffusion_instance,  # noqa: F811
):
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state = savepoint_diffusion_init.construct_prognostics()
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    config = construct_diffusion_config(experiment, ndyn_substeps=2)
    additional_parameters = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion_instance  # the fixture makes sure that the orchestrator cache is cleared properly between pytest runs -if applicable-
    diffusion_granule.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    if linit:
        diffusion_granule.initial_run(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
        )

        verify_diffusion_fields(
            config=config,
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            diffusion_savepoint=savepoint_diffusion_exit,
        )
