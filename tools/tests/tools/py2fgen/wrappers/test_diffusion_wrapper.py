# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
from unittest import mock

import cffi
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import states as grid_states, vertical as v_grid
from icon4py.model.testing import (
    definitions,
    grid_utils,
    serialbox as sb,
    test_utils as testing_test_utils,
)
from icon4py.model.testing.fixtures.datatest import backend
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils
from icon4py.tools.py2fgen.wrappers import common as wrapper_common, diffusion_wrapper

from . import utils
from .test_grid_init import grid_init


@pytest.fixture
def interpolation_state(
    interpolation_savepoint: sb.InterpolationSavepoint,
) -> diffusion_states.DiffusionInterpolationState:
    return diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )


@pytest.fixture
def metric_state(
    metrics_savepoint: sb.MetricSavepoint,
) -> diffusion_states.DiffusionMetricState:
    return diffusion_states.DiffusionMetricState(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
def test_diffusion_wrapper_granule_inputs(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    grid_init,  # initializes the grid as side-effect
    icon_grid,
    experiment,
    ndyn_substeps,
):
    # --- Define Diffusion Configuration ---
    diffusion_type = diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER
    hdiff_w = True
    hdiff_vn = True
    hdiff_temp = True
    hdiff_smag_w = False
    ltkeshs = True
    type_t_diffu = diffusion.TemperatureDiscretizationType.HETEROGENEOUS
    type_vn_diffu = diffusion.SmagorinskyStencilType.DIAMOND_VERTICES
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    zdiffu_t = True
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0
    denom_diffu_v = 150.0
    max_nudging_coefficient = 0.375
    itype_sher = (
        diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND
    )

    # --- Extract Metric State Parameters ---
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)

    # The wrapper expects [cellidx, c2e2c_ids] and then extracts `zd_cellidx[0,:]` because it only needs the cellidxs
    # (this is because slicing causes issue in the bindings, but not for serialization)
    zd_cellidx = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_cellidx", metrics_savepoint.savepoint))[
            np.newaxis, :
        ]
    )
    zd_vertidx = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_vertidx", metrics_savepoint.savepoint))
    )
    zd_intcoef = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_intcoef", metrics_savepoint.savepoint))
    )
    zd_diffcoef = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_diffcoef", metrics_savepoint.savepoint))
    )

    # --- Extract Interpolation State Parameters ---
    e_bln_c_s = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)
    geofac_div = test_utils.array_to_array_info(interpolation_savepoint.geofac_div().ndarray)
    geofac_grg_x_field, geofac_grg_y_field = interpolation_savepoint.geofac_grg()
    geofac_grg_x = test_utils.array_to_array_info(geofac_grg_x_field.ndarray)
    geofac_grg_y = test_utils.array_to_array_info(geofac_grg_y_field.ndarray)
    geofac_n2s = test_utils.array_to_array_info(interpolation_savepoint.geofac_n2s().ndarray)
    nudgecoeff_e = test_utils.array_to_array_info(interpolation_savepoint.nudgecoeff_e().ndarray)
    rbf_coeff_1 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v1().ndarray)
    rbf_coeff_2 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v2().ndarray)

    # --- Extract Diagnostic and Prognostic State Parameters ---
    hdef_ic = test_utils.array_to_array_info(savepoint_diffusion_init.hdef_ic().ndarray)
    div_ic = test_utils.array_to_array_info(savepoint_diffusion_init.div_ic().ndarray)
    dwdx = test_utils.array_to_array_info(savepoint_diffusion_init.dwdx().ndarray)
    dwdy = test_utils.array_to_array_info(savepoint_diffusion_init.dwdy().ndarray)
    w = test_utils.array_to_array_info(savepoint_diffusion_init.w().ndarray)
    vn = test_utils.array_to_array_info(savepoint_diffusion_init.vn().ndarray)
    exner = test_utils.array_to_array_info(savepoint_diffusion_init.exner().ndarray)
    theta_v = test_utils.array_to_array_info(savepoint_diffusion_init.theta_v().ndarray)
    rho = test_utils.array_to_array_info(savepoint_diffusion_init.rho().ndarray)
    dtime = savepoint_diffusion_init.get_metadata("dtime")["dtime"]

    # --- Expected objects that form inputs into init and run functions
    expected_icon_grid = icon_grid
    expected_dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    expected_edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    expected_cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    expected_interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    expected_metric_state = diffusion_states.DiffusionMetricState(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
    expected_diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    expected_prognostic_state = savepoint_diffusion_init.construct_prognostics()
    expected_config = definitions.construct_diffusion_config(experiment, ndyn_substeps)
    expected_additional_parameters = diffusion.DiffusionParams(expected_config)

    # --- Mock and Test Diffusion.init ---
    with mock.patch(
        "icon4py.model.atmosphere.diffusion.diffusion.Diffusion.__init__", return_value=None
    ) as mock_init:
        diffusion_wrapper.diffusion_init(
            ffi=cffi.FFI(),
            perf_counters=None,
            theta_ref_mc=theta_ref_mc,
            wgtfac_c=wgtfac_c,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            geofac_n2s=geofac_n2s,
            nudgecoeff_e=nudgecoeff_e,
            rbf_coeff_1=rbf_coeff_1,
            rbf_coeff_2=rbf_coeff_2,
            zd_cellidx=zd_cellidx,
            zd_vertidx=zd_vertidx,
            zd_intcoef=zd_intcoef,
            zd_diffcoef=zd_diffcoef,
            ndyn_substeps=ndyn_substeps,
            diffusion_type=diffusion_type,
            hdiff_w=hdiff_w,
            hdiff_vn=hdiff_vn,
            hdiff_smag_w=hdiff_smag_w,
            zdiffu_t=zdiffu_t,
            type_t_diffu=type_t_diffu,
            type_vn_diffu=type_vn_diffu,
            hdiff_efdt_ratio=hdiff_efdt_ratio,
            smagorinski_scaling_factor=smagorinski_scaling_factor,
            hdiff_temp=hdiff_temp,
            thslp_zdiffu=thslp_zdiffu,
            thhgtd_zdiffu=thhgtd_zdiffu,
            denom_diffu_v=denom_diffu_v,
            nudge_max_coeff=max_nudging_coefficient,
            itype_sher=itype_sher.value,
            ltkeshs=ltkeshs,
            backend=wrapper_common.BackendIntEnum.DEFAULT,
        )

        # Check input arguments to Diffusion.init
        _, captured_kwargs = mock_init.call_args

        # special case of grid._id as we do not use this arg in the wrapper as we cant pass strings from Fortran to the wrapper
        try:
            result, error_message = utils.compare_objects(
                captured_kwargs["grid"], expected_icon_grid
            )
            assert result, f"Grid comparison failed: {error_message}"
        except AssertionError as e:
            error_message = str(e)
            if "icon_grid != " not in error_message:
                raise
            else:
                pass

        result, error_message = utils.compare_objects(captured_kwargs["config"], expected_config)
        assert result, f"Config comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["params"], expected_additional_parameters
        )
        assert result, f"Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["metric_state"], expected_metric_state
        )
        assert result, f"Metric State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["interpolation_state"], expected_interpolation_state
        )
        assert result, f"Interpolation State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["edge_params"], expected_edge_geometry
        )
        assert result, f"Edge Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["cell_params"], expected_cell_geometry
        )
        assert result, f"Cell Params comparison failed: {error_message}"

    # --- Mock and Test Diffusion.run ---
    with mock.patch("icon4py.model.atmosphere.diffusion.diffusion.Diffusion.run") as mock_run:
        diffusion_wrapper.diffusion_run(
            ffi=cffi.FFI(),
            perf_counters=None,
            w=w,
            vn=vn,
            exner=exner,
            theta_v=theta_v,
            rho=rho,
            hdef_ic=hdef_ic,
            div_ic=div_ic,
            dwdx=dwdx,
            dwdy=dwdy,
            dtime=dtime,
            linit=False,
        )

        # Check input arguments to Diffusion.run
        _, captured_kwargs = mock_run.call_args
        assert utils.compare_objects(captured_kwargs["diagnostic_state"], expected_diagnostic_state)
        assert utils.compare_objects(captured_kwargs["prognostic_state"], expected_prognostic_state)
        assert captured_kwargs["dtime"] == expected_dtime


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        # (
        #     definitions.Experiments.MCH_CH_R04B09,
        #     "2021-06-20T12:00:10.000",
        #     "2021-06-20T12:00:10.000",
        # ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
#@pytest.mark.parametrize("ndyn_substeps", (2,))
@pytest.mark.parametrize("ndyn_substeps", [2])
#@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
@pytest.mark.parametrize("orchestration", [False])
def test_diffusion_wrapper_single_step(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    grid_init,  # initializes the grid as side-effect
    experiment,
    ndyn_substeps,
    step_date_init,
    step_date_exit,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    backend,
    orchestration,
):
    config = definitions.construct_diffusion_config(experiment, ndyn_substeps)
    diffusion_type = config.diffusion_type
    hdiff_w = config.apply_to_vertical_wind
    hdiff_vn = config.apply_to_horizontal_wind
    hdiff_temp = config.apply_to_temperature
    hdiff_smag_w = config.apply_smag_diff_to_vertical_wind
    ltkeshs = config.ltkeshs
    type_t_diffu = config.type_t_diffu
    type_vn_diffu = config.type_vn_diffu
    hdiff_efdt_ratio = config.hdiff_efdt_ratio
    smagorinski_scaling_factor = config.smagorinski_scaling_factor
    zdiffu_t = config.apply_zdiffusion_t
    thslp_zdiffu = config.thslp_zdiffu
    thhgtd_zdiffu = config.thhgtd_zdiffu
    denom_diffu_v = config.velocity_boundary_diffusion_denominator
    max_nudging_coefficient = config.max_nudging_coefficient
    itype_sher = config.shear_type

    # Metric state parameters
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)

    if zdiffu_t:
        # The wrapper expects [cellidx, c2e2c_ids] and then extracts `zd_cellidx[0,:]` because it only needs the cellidxs
        # (this is because slicing causes issue in the bindings, but not for serialization)
        zd_cellidx = test_utils.array_to_array_info(
            np.squeeze(metrics_savepoint.serializer.read("zd_cellidx", metrics_savepoint.savepoint))[
                np.newaxis, :
            ]
        )
        zd_vertidx = test_utils.array_to_array_info(
            np.squeeze(metrics_savepoint.serializer.read("zd_vertidx", metrics_savepoint.savepoint))
        )
        zd_intcoef = test_utils.array_to_array_info(
            np.squeeze(metrics_savepoint.serializer.read("zd_intcoef", metrics_savepoint.savepoint))
        )
        zd_diffcoef = test_utils.array_to_array_info(
            np.squeeze(metrics_savepoint.serializer.read("zd_diffcoef", metrics_savepoint.savepoint))
        )
    else:
        zd_cellidx = zd_vertidx = zd_intcoef = zd_diffcoef = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)  # dummy values as they are not used when zdiffu_t is False

    # Interpolation state parameters
    e_bln_c_s = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)
    geofac_div = test_utils.array_to_array_info(interpolation_savepoint.geofac_div().ndarray)
    geofac_grg_x_field, geofac_grg_y_field = interpolation_savepoint.geofac_grg()
    geofac_grg_x = test_utils.array_to_array_info(geofac_grg_x_field.ndarray)
    geofac_grg_y = test_utils.array_to_array_info(geofac_grg_y_field.ndarray)
    geofac_n2s = test_utils.array_to_array_info(interpolation_savepoint.geofac_n2s().ndarray)
    nudgecoeff_e = test_utils.array_to_array_info(interpolation_savepoint.nudgecoeff_e().ndarray)
    rbf_coeff_1 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v1().ndarray)
    rbf_coeff_2 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v2().ndarray)

    # Diagnostic state parameters
    hdef_ic = test_utils.array_to_array_info(savepoint_diffusion_init.hdef_ic().ndarray)
    div_ic = test_utils.array_to_array_info(savepoint_diffusion_init.div_ic().ndarray)
    dwdx = test_utils.array_to_array_info(savepoint_diffusion_init.dwdx().ndarray)
    dwdy = test_utils.array_to_array_info(savepoint_diffusion_init.dwdy().ndarray)

    # Prognostic state parameters
    w = test_utils.array_to_array_info(savepoint_diffusion_init.w().ndarray)
    vn = test_utils.array_to_array_info(savepoint_diffusion_init.vn().ndarray)
    exner = test_utils.array_to_array_info(savepoint_diffusion_init.exner().ndarray)
    theta_v = test_utils.array_to_array_info(savepoint_diffusion_init.theta_v().ndarray)
    rho = test_utils.array_to_array_info(savepoint_diffusion_init.rho().ndarray)
    dtime = savepoint_diffusion_init.get_metadata("dtime")["dtime"]

    ffi = cffi.FFI()
    # Call diffusion_init
    diffusion_wrapper.diffusion_init(
        ffi=ffi,
        perf_counters=None,
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        e_bln_c_s=e_bln_c_s,
        geofac_div=geofac_div,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        geofac_n2s=geofac_n2s,
        nudgecoeff_e=nudgecoeff_e,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        zd_cellidx=zd_cellidx,
        zd_vertidx=zd_vertidx,
        zd_intcoef=zd_intcoef,
        zd_diffcoef=zd_diffcoef,
        ndyn_substeps=ndyn_substeps,
        diffusion_type=diffusion_type,
        hdiff_w=hdiff_w,
        hdiff_vn=hdiff_vn,
        hdiff_smag_w=hdiff_smag_w,
        zdiffu_t=zdiffu_t,
        type_t_diffu=type_t_diffu,
        type_vn_diffu=type_vn_diffu,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        smagorinski_scaling_factor=smagorinski_scaling_factor,
        hdiff_temp=hdiff_temp,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        denom_diffu_v=denom_diffu_v,
        nudge_max_coeff=max_nudging_coefficient,
        itype_sher=itype_sher.value,
        ltkeshs=ltkeshs,
        #backend=wrapper_common.BackendIntEnum.DEFAULT,
        backend=wrapper_common.BackendIntEnum.GTFN,
    )

    # Call diffusion_run
    diffusion_wrapper.diffusion_run(
        ffi=ffi,
        perf_counters=None,
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
        dtime=dtime,
        linit=False,
    )

    grid_functionality = collections.defaultdict(dict)
    def get_grid_for_experiment(experiment: definitions.Experiment, backend: gtx_typing.Backend):
        return _get_or_initialize(experiment, backend, "grid")
    def get_edge_geometry_for_experiment(
        experiment: definitions.Experiment, backend: gtx_typing.Backend
    ):
        return _get_or_initialize(experiment, backend, "edge_geometry")
    def get_cell_geometry_for_experiment(
        experiment: definitions.Experiment, backend: gtx_typing.Backend
    ):
        return _get_or_initialize(experiment, backend, "cell_geometry")
    def _get_or_initialize(experiment: definitions.Experiment, backend: gtx_typing.Backend, name: str):
        from icon4py.model.common.grid import geometry_attributes as geometry_meta
        if not grid_functionality[experiment.name].get(name):
            geometry_ = grid_utils.get_grid_geometry(backend, experiment)
            grid = geometry_.grid
            cell_params = grid_states.CellParams(
                cell_center_lat=geometry_.get(geometry_meta.CELL_LAT),
                cell_center_lon=geometry_.get(geometry_meta.CELL_LON),
                area=geometry_.get(geometry_meta.CELL_AREA),
            )
            edge_params = grid_states.EdgeParams(
                edge_center_lat=geometry_.get(geometry_meta.EDGE_LAT),
                edge_center_lon=geometry_.get(geometry_meta.EDGE_LON),
                tangent_orientation=geometry_.get(geometry_meta.TANGENT_ORIENTATION),
                coriolis_frequency=geometry_.get(geometry_meta.CORIOLIS_PARAMETER),
                edge_areas=geometry_.get(geometry_meta.EDGE_AREA),
                primal_edge_lengths=geometry_.get(geometry_meta.EDGE_LENGTH),
                inverse_primal_edge_lengths=geometry_.get(f"inverse_of_{geometry_meta.EDGE_LENGTH}"),
                dual_edge_lengths=geometry_.get(geometry_meta.DUAL_EDGE_LENGTH),
                inverse_dual_edge_lengths=geometry_.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"),
                inverse_vertex_vertex_lengths=geometry_.get(
                    f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
                ),
                primal_normal_x=geometry_.get(geometry_meta.EDGE_NORMAL_U),
                primal_normal_y=geometry_.get(geometry_meta.EDGE_NORMAL_V),
                primal_normal_cell_x=geometry_.get(geometry_meta.EDGE_NORMAL_CELL_U),
                primal_normal_cell_y=geometry_.get(geometry_meta.EDGE_NORMAL_CELL_V),
                primal_normal_vert_x=geometry_.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
                primal_normal_vert_y=geometry_.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
                dual_normal_cell_x=geometry_.get(geometry_meta.EDGE_TANGENT_CELL_U),
                dual_normal_cell_y=geometry_.get(geometry_meta.EDGE_TANGENT_CELL_V),
                dual_normal_vert_x=geometry_.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
                dual_normal_vert_y=geometry_.get(geometry_meta.EDGE_TANGENT_VERTEX_V),
            )
            grid_functionality[experiment.name]["grid"] = grid
            grid_functionality[experiment.name]["edge_geometry"] = edge_params
            grid_functionality[experiment.name]["cell_geometry"] = cell_params
        return grid_functionality[experiment.name].get(name)
    grid = get_grid_for_experiment(experiment, backend)
    cell_geometry = get_cell_geometry_for_experiment(experiment, backend)
    edge_geometry = get_edge_geometry_for_experiment(experiment, backend)
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state = savepoint_diffusion_init.construct_prognostics()
    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    #config = definitions.construct_diffusion_config(experiment, ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)
    print(f"damping_height: {damping_height}, stretch_factor: {stretch_factor}, model_top_height: {model_top_height}, lowest_layer_thickness: {lowest_layer_thickness}")
    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        orchestration=orchestration,
    )
    diffusion_granule.run(
        diagnostic_state=diagnostic_state, prognostic_state=prognostic_state, dtime=dtime
    )

    # Assertions comparing the serialized output with computed output fields
    w_ = savepoint_diffusion_exit.w()
    vn_ = savepoint_diffusion_exit.vn()
    exner_ = savepoint_diffusion_exit.exner()
    theta_v_ = savepoint_diffusion_exit.theta_v()
    if itype_sher > 0:
        hdef_ic_ = savepoint_diffusion_exit.hdef_ic()
        div_ic_ = savepoint_diffusion_exit.div_ic()
    if itype_sher == diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND:
        dwdx_ = savepoint_diffusion_exit.dwdx()
        dwdy_ = savepoint_diffusion_exit.dwdy()

    print(f"max(abs(w): {np.max(np.abs(w_.asnumpy() - py2fgen.as_array(ffi, w, py2fgen.FLOAT64)))}")
    print(f"max(abs(vn): {np.max(np.abs(vn_.asnumpy() - py2fgen.as_array(ffi, vn, py2fgen.FLOAT64)))}")
    print(f"max(abs(exner): {np.max(np.abs(exner_.asnumpy() - py2fgen.as_array(ffi, exner, py2fgen.FLOAT64)))}")
    print(f"max(abs(theta_v): {np.max(np.abs(theta_v_.asnumpy() - py2fgen.as_array(ffi, theta_v, py2fgen.FLOAT64)))}")

    print(f"max(abs(w) granule: {np.max(np.abs(prognostic_state.w.asnumpy() - py2fgen.as_array(ffi, w, py2fgen.FLOAT64)))}")
    print(f"max(abs(vn) granule: {np.max(np.abs(prognostic_state.vn.asnumpy() - py2fgen.as_array(ffi, vn, py2fgen.FLOAT64)))}")

    print(f"max(abs(w) granule_v: {np.max(np.abs(prognostic_state.w.asnumpy() - w_.asnumpy()))}")

    breakpoint()

    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, w, py2fgen.FLOAT64), w_.asnumpy(), atol=1e-12
    )
    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, vn, py2fgen.FLOAT64), vn_.asnumpy(), atol=1e-12
    )
    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, exner, py2fgen.FLOAT64), exner_.asnumpy(), atol=1e-12
    )
    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, theta_v, py2fgen.FLOAT64), theta_v_.asnumpy(), atol=1e-12
    )
    if itype_sher > 0:
        assert testing_test_utils.dallclose(
            py2fgen.as_array(ffi, hdef_ic, py2fgen.FLOAT64), hdef_ic_.asnumpy(), atol=1e-12
        )
        assert testing_test_utils.dallclose(
            py2fgen.as_array(ffi, div_ic, py2fgen.FLOAT64), div_ic_.asnumpy(), atol=1e-12
        )
    if itype_sher == diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND:
        assert testing_test_utils.dallclose(
            py2fgen.as_array(ffi, dwdx, py2fgen.FLOAT64), dwdx_.asnumpy(), atol=1e-12
        )
        assert testing_test_utils.dallclose(
            py2fgen.as_array(ffi, dwdy, py2fgen.FLOAT64), dwdy_.asnumpy(), atol=1e-12
        )
