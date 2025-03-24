# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import cffi
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import states as grid_states, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils
from icon4py.tools.py2fgen.wrappers import (
    common as wrapper_common,
    diffusion_wrapper,
)

from . import utils
from .test_grid_init import grid_init  # noqa: F401


@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_diffusion_wrapper_granule_inputs(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    grid_init,  # noqa: F811  # initializes the grid as side-effect
    icon_grid,
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
):
    # --- Define Diffusion Configuration ---
    diffusion_type = diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER
    hdiff_w = True
    hdiff_vn = True
    hdiff_temp = True
    ltkeshs = True
    type_t_diffu = 2
    type_vn_diffu = 1
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    zdiffu_t = True
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0
    denom_diffu_v = 150.0
    nudge_max_coeff = 0.075 * constants.DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
    itype_sher = (
        diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND
    )
    nflat_gradp = grid_savepoint.nflat_gradp()

    # --- Extract Metric State Parameters ---
    vct_a = test_utils.array_to_array_info(grid_savepoint.vct_a().ndarray)
    vct_b = test_utils.array_to_array_info(grid_savepoint.vct_b().ndarray)
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)
    mask_hdiff = test_utils.array_to_array_info(metrics_savepoint.mask_hdiff().ndarray)
    zd_diffcoef = test_utils.array_to_array_info(metrics_savepoint.zd_diffcoef().ndarray)

    # todo: special handling, determine if this is necessary for Fortran arrays too
    zd_vertoffset = np.squeeze(
        metrics_savepoint.serializer.read("zd_vertoffset", metrics_savepoint.savepoint)
    )
    zd_vertoffset = metrics_savepoint._reduce_to_dim_size(
        zd_vertoffset, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_vertoffset = test_utils.array_to_array_info(zd_vertoffset)

    zd_intcoef = np.squeeze(metrics_savepoint.serializer.read("vcoef", metrics_savepoint.savepoint))
    zd_intcoef = metrics_savepoint._reduce_to_dim_size(
        zd_intcoef, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_intcoef = test_utils.array_to_array_info(zd_intcoef)

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
        e_bln_c_s=data_alloc.flatten_first_two_dims(
            dims.CEDim, field=interpolation_savepoint.e_bln_c_s()
        ),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=data_alloc.flatten_first_two_dims(
            dims.CEDim, field=interpolation_savepoint.geofac_div()
        ),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    expected_metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_savepoint.mask_hdiff(),
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
    expected_vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    expected_vertical_params = v_grid.VerticalGrid(
        config=expected_vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    expected_config = utils.construct_diffusion_config(experiment, ndyn_substeps)
    expected_additional_parameters = diffusion.DiffusionParams(expected_config)

    # --- Mock and Test Diffusion.init ---
    with mock.patch(
        "icon4py.model.atmosphere.diffusion.diffusion.Diffusion.__init__", return_value=None
    ) as mock_init:
        diffusion_wrapper.diffusion_init(
            ffi=cffi.FFI(),
            meta=None,
            vct_a=vct_a,
            vct_b=vct_b,
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
            mask_hdiff=mask_hdiff,
            zd_diffcoef=zd_diffcoef,
            zd_vertoffset=zd_vertoffset,
            zd_intcoef=zd_intcoef,
            ndyn_substeps=ndyn_substeps,
            rayleigh_damping_height=damping_height,
            nflat_gradp=nflat_gradp,
            diffusion_type=diffusion_type,
            hdiff_w=hdiff_w,
            hdiff_vn=hdiff_vn,
            zdiffu_t=zdiffu_t,
            type_t_diffu=type_t_diffu,
            type_vn_diffu=type_vn_diffu,
            hdiff_efdt_ratio=hdiff_efdt_ratio,
            smagorinski_scaling_factor=smagorinski_scaling_factor,
            hdiff_temp=hdiff_temp,
            thslp_zdiffu=thslp_zdiffu,
            thhgtd_zdiffu=thhgtd_zdiffu,
            denom_diffu_v=denom_diffu_v,
            nudge_max_coeff=nudge_max_coeff,
            itype_sher=itype_sher.value,
            ltkeshs=ltkeshs,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            backend=wrapper_common.BackendIntEnum.DEFAULT,
        )

        # Check input arguments to Diffusion.init
        captured_args, captured_kwargs = mock_init.call_args

        # special case of grid._id as we do not use this arg in the wrapper as we cant pass strings from Fortran to the wrapper
        try:
            result, error_message = utils.compare_objects(
                captured_kwargs["grid"], expected_icon_grid
            )
            assert result, f"Grid comparison failed: {error_message}"
        except AssertionError as e:
            error_message = str(e)
            if "object.connectivities" not in error_message:
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
            captured_kwargs["vertical_grid"], expected_vertical_params
        )
        assert result, f"Vertical Grid comparison failed: {error_message}"

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
            meta=None,
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
        captured_args, captured_kwargs = mock_run.call_args
        assert utils.compare_objects(captured_kwargs["diagnostic_state"], expected_diagnostic_state)
        assert utils.compare_objects(captured_kwargs["prognostic_state"], expected_prognostic_state)
        assert captured_kwargs["dtime"] == expected_dtime


@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_diffusion_wrapper_single_step(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    grid_init,  # noqa: F811  # initializes the grid as side-effect
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    step_date_init,
    step_date_exit,
):
    # Hardcoded DiffusionConfig parameters
    diffusion_type = diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER
    hdiff_w = True
    hdiff_vn = True
    hdiff_temp = True
    ltkeshs = True
    type_t_diffu = 2
    type_vn_diffu = 1
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    zdiffu_t = True
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0
    denom_diffu_v = 150.0
    nudge_max_coeff = (
        0.075 * constants.DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
    )  # this is done in ICON, so we replicate it here
    itype_sher = (
        diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND
    )
    nflat_gradp = grid_savepoint.nflat_gradp()

    # Metric state parameters
    vct_a = test_utils.array_to_array_info(grid_savepoint.vct_a().ndarray)
    vct_b = test_utils.array_to_array_info(grid_savepoint.vct_b().ndarray)
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)
    mask_hdiff = test_utils.array_to_array_info(metrics_savepoint.mask_hdiff().ndarray)
    zd_diffcoef = test_utils.array_to_array_info(metrics_savepoint.zd_diffcoef().ndarray)

    # todo: special handling, determine if this is necessary for Fortran arrays too
    zd_vertoffset = np.squeeze(
        metrics_savepoint.serializer.read("zd_vertoffset", metrics_savepoint.savepoint)
    )
    zd_vertoffset = metrics_savepoint._reduce_to_dim_size(
        zd_vertoffset, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_vertoffset = test_utils.array_to_array_info(zd_vertoffset)

    zd_intcoef = np.squeeze(metrics_savepoint.serializer.read("vcoef", metrics_savepoint.savepoint))
    zd_intcoef = metrics_savepoint._reduce_to_dim_size(
        zd_intcoef, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_intcoef = test_utils.array_to_array_info(zd_intcoef)

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
        meta=None,
        vct_a=vct_a,
        vct_b=vct_b,
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
        mask_hdiff=mask_hdiff,
        zd_diffcoef=zd_diffcoef,
        zd_vertoffset=zd_vertoffset,
        zd_intcoef=zd_intcoef,
        ndyn_substeps=ndyn_substeps,
        rayleigh_damping_height=damping_height,
        nflat_gradp=nflat_gradp,
        diffusion_type=diffusion_type,
        hdiff_w=hdiff_w,
        hdiff_vn=hdiff_vn,
        zdiffu_t=zdiffu_t,
        type_t_diffu=type_t_diffu,
        type_vn_diffu=type_vn_diffu,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        smagorinski_scaling_factor=smagorinski_scaling_factor,
        hdiff_temp=hdiff_temp,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        denom_diffu_v=denom_diffu_v,
        nudge_max_coeff=nudge_max_coeff,
        itype_sher=itype_sher.value,
        ltkeshs=ltkeshs,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        backend=wrapper_common.BackendIntEnum.DEFAULT,
    )

    # Call diffusion_run
    diffusion_wrapper.diffusion_run(
        ffi=ffi,
        meta=None,
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

    # Assertions comparing the serialized output with computed output fields
    w_ = savepoint_diffusion_exit.w()
    vn_ = savepoint_diffusion_exit.vn()
    exner_ = savepoint_diffusion_exit.exner()
    theta_v_ = savepoint_diffusion_exit.theta_v()
    hdef_ic_ = savepoint_diffusion_exit.hdef_ic()
    div_ic_ = savepoint_diffusion_exit.div_ic()
    dwdx_ = savepoint_diffusion_exit.dwdx()
    dwdy_ = savepoint_diffusion_exit.dwdy()

    assert helpers.dallclose(py2fgen.as_array(ffi, w, py2fgen.FLOAT64), w_.asnumpy(), atol=1e-12)
    assert helpers.dallclose(py2fgen.as_array(ffi, vn, py2fgen.FLOAT64), vn_.asnumpy(), atol=1e-12)
    assert helpers.dallclose(
        py2fgen.as_array(ffi, exner, py2fgen.FLOAT64), exner_.asnumpy(), atol=1e-12
    )
    assert helpers.dallclose(
        py2fgen.as_array(ffi, theta_v, py2fgen.FLOAT64), theta_v_.asnumpy(), atol=1e-12
    )
    assert helpers.dallclose(
        py2fgen.as_array(ffi, hdef_ic, py2fgen.FLOAT64), hdef_ic_.asnumpy(), atol=1e-12
    )
    assert helpers.dallclose(
        py2fgen.as_array(ffi, div_ic, py2fgen.FLOAT64), div_ic_.asnumpy(), atol=1e-12
    )
    assert helpers.dallclose(
        py2fgen.as_array(ffi, dwdx, py2fgen.FLOAT64), dwdx_.asnumpy(), atol=1e-12
    )
    assert helpers.dallclose(
        py2fgen.as_array(ffi, dwdy, py2fgen.FLOAT64), dwdy_.asnumpy(), atol=1e-12
    )
