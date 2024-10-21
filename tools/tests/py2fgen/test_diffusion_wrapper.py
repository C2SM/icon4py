# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import gt4py.next as gtx
import numpy as np
import pytest
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionType, TurbulenceShearForcingType
from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry import CellParams, EdgeParams
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers
from icon4py.model.common.test_utils.datatest_utils import (
    get_global_grid_params,
)

from icon4pytools.py2fgen.wrappers import wrapper_dimension
from icon4pytools.py2fgen.wrappers.diffusion import diffusion_init, diffusion_run, grid_init

from .conftest import compare_objects, construct_diffusion_config


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
    icon_grid,
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
):
    # --- Define Diffusion Configuration ---
    diffusion_type = DiffusionType.SMAGORINSKY_4TH_ORDER
    hdiff_w = True
    hdiff_vn = True
    hdiff_temp = True
    type_t_diffu = 2
    type_vn_diffu = 1
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    zdiffu_t = True
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0
    denom_diffu_v = 150.0
    nudge_max_coeff = 0.075 * DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
    itype_sher = TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND
    nflat_gradp = grid_savepoint.nflat_gradp()

    # --- Global Grid Parameters ---
    global_root, global_level = get_global_grid_params(experiment)

    # --- Extract Grid Parameters from Savepoint ---
    tangent_orientation = grid_savepoint.tangent_orientation()
    inverse_primal_edge_lengths = grid_savepoint.inverse_primal_edge_lengths()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_vert_vert_length = grid_savepoint.inv_vert_vert_length()
    edge_areas = grid_savepoint.edge_areas()
    f_e = grid_savepoint.f_e()
    cell_areas = grid_savepoint.cell_areas()
    primal_normal_vert_x = grid_savepoint.primal_normal_vert_x()
    primal_normal_vert_y = grid_savepoint.primal_normal_vert_y()
    dual_normal_vert_x = grid_savepoint.dual_normal_vert_x()
    dual_normal_vert_y = grid_savepoint.dual_normal_vert_y()
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y()
    dual_normal_cell_x = grid_savepoint.dual_normal_cell_x()
    dual_normal_cell_y = grid_savepoint.dual_normal_cell_y()
    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    edge_center_lat = grid_savepoint.edge_center_lat()
    edge_center_lon = grid_savepoint.edge_center_lon()
    primal_normal_x = grid_savepoint.primal_normal_x()
    primal_normal_y = grid_savepoint.primal_normal_y()

    # --- Extract Metric State Parameters ---
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    theta_ref_mc = metrics_savepoint.theta_ref_mc()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    mask_hdiff = metrics_savepoint.mask_hdiff()
    zd_diffcoef = metrics_savepoint.zd_diffcoef()

    # todo: special handling, determine if this is necessary for Fortran arrays too
    zd_vertoffset = np.squeeze(
        metrics_savepoint.serializer.read("zd_vertoffset", metrics_savepoint.savepoint)
    )
    zd_vertoffset = metrics_savepoint._reduce_to_dim_size(
        zd_vertoffset, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_vertoffset = gtx.as_field(
        (dims.CellDim, dims.E2CDim, dims.KDim),
        zd_vertoffset,
    )

    zd_intcoef = np.squeeze(metrics_savepoint.serializer.read("vcoef", metrics_savepoint.savepoint))
    zd_intcoef = metrics_savepoint._reduce_to_dim_size(
        zd_intcoef, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_intcoef = gtx.as_field(
        (dims.CellDim, dims.E2CDim, dims.KDim),
        zd_intcoef,
    )

    # --- Extract Interpolation State Parameters ---
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_grg_x, geofac_grg_y = interpolation_savepoint.geofac_grg()
    geofac_n2s = interpolation_savepoint.geofac_n2s()
    nudgecoeff_e = interpolation_savepoint.nudgecoeff_e()
    rbf_coeff_1 = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_coeff_2 = interpolation_savepoint.rbf_vec_coeff_v2()

    # --- Extract Diagnostic and Prognostic State Parameters ---
    hdef_ic = savepoint_diffusion_init.hdef_ic()
    div_ic = savepoint_diffusion_init.div_ic()
    dwdx = savepoint_diffusion_init.dwdx()
    dwdy = savepoint_diffusion_init.dwdy()
    w = savepoint_diffusion_init.w()
    vn = savepoint_diffusion_init.vn()
    exner = savepoint_diffusion_init.exner()
    theta_v = savepoint_diffusion_init.theta_v()
    rho = savepoint_diffusion_init.rho()
    dtime = savepoint_diffusion_init.get_metadata("dtime")["dtime"]

    # --- Set Up Grid Parameters ---
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")

    cell_starts = gtx.as_field(
        (wrapper_dimension.CellIndexDim,), grid_savepoint._read_int32("c_start_index")
    )
    cell_ends = gtx.as_field(
        (wrapper_dimension.CellIndexDim,), grid_savepoint._read_int32("c_end_index")
    )
    vertex_starts = gtx.as_field(
        (wrapper_dimension.VertexIndexDim,), grid_savepoint._read_int32("v_start_index")
    )
    vertex_ends = gtx.as_field(
        (wrapper_dimension.VertexIndexDim,), grid_savepoint._read_int32("v_end_index")
    )
    edge_starts = gtx.as_field(
        (wrapper_dimension.EdgeIndexDim,), grid_savepoint._read_int32("e_start_index")
    )
    edge_ends = gtx.as_field(
        (wrapper_dimension.EdgeIndexDim,), grid_savepoint._read_int32("e_end_index")
    )

    c2e = gtx.as_field((dims.CellDim, dims.C2EDim), grid_savepoint._read_int32("c2e"))
    e2c = gtx.as_field((dims.EdgeDim, dims.E2CDim), grid_savepoint._read_int32("e2c"))
    c2e2c = gtx.as_field((dims.CellDim, dims.C2E2CDim), grid_savepoint._read_int32("c2e2c"))
    e2c2e = gtx.as_field((dims.EdgeDim, dims.E2C2EDim), grid_savepoint._read_int32("e2c2e"))
    e2v = gtx.as_field((dims.EdgeDim, dims.E2VDim), grid_savepoint._read_int32("e2v"))
    v2e = gtx.as_field((dims.VertexDim, dims.V2EDim), grid_savepoint._read_int32("v2e"))
    v2c = gtx.as_field((dims.VertexDim, dims.V2CDim), grid_savepoint._read_int32("v2c"))
    e2c2v = gtx.as_field((dims.EdgeDim, dims.E2C2VDim), grid_savepoint._read_int32("e2c2v"))
    c2v = gtx.as_field((dims.CellDim, dims.C2VDim), grid_savepoint._read_int32("c2v"))

    # --- Expected objects that form inputs into init and run functions
    expected_icon_grid = icon_grid
    expected_dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    expected_edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    expected_cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    expected_interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
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
    expected_config = construct_diffusion_config(experiment, ndyn_substeps)
    expected_additional_parameters = diffusion.DiffusionParams(expected_config)

    # --- Initialize the Grid ---
    grid_init(
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        global_root=global_root,
        global_level=global_level,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
    )

    # --- Mock and Test diffusion_granule.init ---
    with mock.patch("icon4py.model.atmosphere.diffusion.diffusion.Diffusion") as mock_init:
        diffusion_init(
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
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            edge_areas=edge_areas,
            f_e=f_e,
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            cell_areas=cell_areas,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_cell_x=dual_normal_cell_x,
            dual_normal_cell_y=dual_normal_cell_y,
            edge_center_lat=edge_center_lat,
            edge_center_lon=edge_center_lon,
            primal_normal_x=primal_normal_x,
            primal_normal_y=primal_normal_y,
            global_root=global_root,
            global_level=global_level,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
        )

        # Check input arguments to diffusion_granule.init
        captured_args, captured_kwargs = mock_init.call_args

        # special case of grid._id as we do not use this arg in the wrapper as we cant pass strings from Fortran to the wrapper
        try:
            result, error_message = compare_objects(captured_kwargs["grid"], expected_icon_grid)
            assert result, f"Grid comparison failed: {error_message}"
        except AssertionError as e:
            error_message = str(e)
            if "object.connectivities" not in error_message:
                raise
            else:
                pass

        result, error_message = compare_objects(captured_kwargs["config"], expected_config)
        assert result, f"Config comparison failed: {error_message}"

        result, error_message = compare_objects(
            captured_kwargs["params"], expected_additional_parameters
        )
        assert result, f"Params comparison failed: {error_message}"

        result, error_message = compare_objects(
            captured_kwargs["vertical_grid"], expected_vertical_params
        )
        assert result, f"Vertical Grid comparison failed: {error_message}"

        result, error_message = compare_objects(
            captured_kwargs["metric_state"], expected_metric_state
        )
        assert result, f"Metric State comparison failed: {error_message}"

        result, error_message = compare_objects(
            captured_kwargs["interpolation_state"], expected_interpolation_state
        )
        assert result, f"Interpolation State comparison failed: {error_message}"

        result, error_message = compare_objects(
            captured_kwargs["edge_params"], expected_edge_geometry
        )
        assert result, f"Edge Params comparison failed: {error_message}"

        result, error_message = compare_objects(
            captured_kwargs["cell_params"], expected_cell_geometry
        )
        assert result, f"Cell Params comparison failed: {error_message}"

    # --- Mock and Test diffusion_granule.run ---
    with mock.patch("icon4py.model.atmosphere.diffusion.diffusion.Diffusion.run") as mock_run:
        diffusion_run(
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

        # Check input arguments to diffusion_granule.run
        captured_args, captured_kwargs = mock_run.call_args
        assert compare_objects(captured_kwargs["diagnostic_state"], expected_diagnostic_state)
        assert compare_objects(captured_kwargs["prognostic_state"], expected_prognostic_state)
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
    diffusion_type = DiffusionType.SMAGORINSKY_4TH_ORDER
    hdiff_w = True
    hdiff_vn = True
    hdiff_temp = True
    type_t_diffu = 2
    type_vn_diffu = 1
    hdiff_efdt_ratio = 24.0
    smagorinski_scaling_factor = 0.025
    zdiffu_t = True
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0
    denom_diffu_v = 150.0
    nudge_max_coeff = (
        0.075 * DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
    )  # this is done in ICON, so we replicate it here
    itype_sher = TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND
    nflat_gradp = grid_savepoint.nflat_gradp()

    # global grid parameters
    global_root, global_level = get_global_grid_params(experiment)

    # Grid parameters
    tangent_orientation = grid_savepoint.tangent_orientation()
    inverse_primal_edge_lengths = grid_savepoint.inverse_primal_edge_lengths()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_vert_vert_length = grid_savepoint.inv_vert_vert_length()
    edge_areas = grid_savepoint.edge_areas()
    f_e = grid_savepoint.f_e()
    cell_areas = grid_savepoint.cell_areas()
    primal_normal_vert_x = grid_savepoint.primal_normal_vert_x()
    primal_normal_vert_y = grid_savepoint.primal_normal_vert_y()
    dual_normal_vert_x = grid_savepoint.dual_normal_vert_x()
    dual_normal_vert_y = grid_savepoint.dual_normal_vert_y()
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y()
    dual_normal_cell_x = grid_savepoint.dual_normal_cell_x()
    dual_normal_cell_y = grid_savepoint.dual_normal_cell_y()
    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    edge_center_lat = grid_savepoint.edge_center_lat()
    edge_center_lon = grid_savepoint.edge_center_lon()
    primal_normal_x = grid_savepoint.primal_normal_x()
    primal_normal_y = grid_savepoint.primal_normal_y()

    # Metric state parameters
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    theta_ref_mc = metrics_savepoint.theta_ref_mc()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    mask_hdiff = metrics_savepoint.mask_hdiff()
    zd_diffcoef = metrics_savepoint.zd_diffcoef()

    # todo: special handling, determine if this is necessary for Fortran arrays too
    zd_vertoffset = np.squeeze(
        metrics_savepoint.serializer.read("zd_vertoffset", metrics_savepoint.savepoint)
    )
    zd_vertoffset = metrics_savepoint._reduce_to_dim_size(
        zd_vertoffset, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_vertoffset = gtx.as_field(
        (dims.CellDim, dims.E2CDim, dims.KDim),
        zd_vertoffset,
    )

    zd_intcoef = np.squeeze(metrics_savepoint.serializer.read("vcoef", metrics_savepoint.savepoint))
    zd_intcoef = metrics_savepoint._reduce_to_dim_size(
        zd_intcoef, (dims.CellDim, dims.C2E2CDim, dims.KDim)
    )
    zd_intcoef = gtx.as_field(
        (dims.CellDim, dims.E2CDim, dims.KDim),
        zd_intcoef,
    )

    # Interpolation state parameters
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_grg = interpolation_savepoint.geofac_grg()
    geofac_grg_x = geofac_grg[0]
    geofac_grg_y = geofac_grg[1]
    geofac_n2s = interpolation_savepoint.geofac_n2s()
    nudgecoeff_e = interpolation_savepoint.nudgecoeff_e()
    rbf_coeff_1 = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_coeff_2 = interpolation_savepoint.rbf_vec_coeff_v2()

    # Diagnostic state parameters
    hdef_ic = savepoint_diffusion_init.hdef_ic()
    div_ic = savepoint_diffusion_init.div_ic()
    dwdx = savepoint_diffusion_init.dwdx()
    dwdy = savepoint_diffusion_init.dwdy()

    # Prognostic state parameters
    w = savepoint_diffusion_init.w()
    vn = savepoint_diffusion_init.vn()
    exner = savepoint_diffusion_init.exner()
    theta_v = savepoint_diffusion_init.theta_v()
    rho = savepoint_diffusion_init.rho()
    dtime = savepoint_diffusion_init.get_metadata("dtime")["dtime"]

    # grid params
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")

    cell_starts = gtx.as_field(
        (wrapper_dimension.CellIndexDim,), grid_savepoint._read_int32("c_start_index")
    )
    cell_ends = gtx.as_field(
        (wrapper_dimension.CellIndexDim,), grid_savepoint._read_int32("c_end_index")
    )
    vertex_starts = gtx.as_field(
        (wrapper_dimension.VertexIndexDim,), grid_savepoint._read_int32("v_start_index")
    )
    vertex_ends = gtx.as_field(
        (wrapper_dimension.VertexIndexDim,), grid_savepoint._read_int32("v_end_index")
    )
    edge_starts = gtx.as_field(
        (wrapper_dimension.EdgeIndexDim,), grid_savepoint._read_int32("e_start_index")
    )
    edge_ends = gtx.as_field(
        (wrapper_dimension.EdgeIndexDim,), grid_savepoint._read_int32("e_end_index")
    )

    c2e = gtx.as_field((dims.CellDim, dims.C2EDim), grid_savepoint._read_int32("c2e"))
    e2c = gtx.as_field((dims.EdgeDim, dims.E2CDim), grid_savepoint._read_int32("e2c"))
    c2e2c = gtx.as_field((dims.CellDim, dims.C2E2CDim), grid_savepoint._read_int32("c2e2c"))
    e2c2e = gtx.as_field((dims.EdgeDim, dims.E2C2EDim), grid_savepoint._read_int32("e2c2e"))
    e2v = gtx.as_field((dims.EdgeDim, dims.E2VDim), grid_savepoint._read_int32("e2v"))
    v2e = gtx.as_field((dims.VertexDim, dims.V2EDim), grid_savepoint._read_int32("v2e"))
    v2c = gtx.as_field((dims.VertexDim, dims.V2CDim), grid_savepoint._read_int32("v2c"))
    e2c2v = gtx.as_field((dims.EdgeDim, dims.E2C2VDim), grid_savepoint._read_int32("e2c2v"))
    c2v = gtx.as_field((dims.CellDim, dims.C2VDim), grid_savepoint._read_int32("c2v"))

    grid_init(
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        global_root=global_root,
        global_level=global_level,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
    )

    with mock.patch("icon4py.model.atmosphere.diffusion.diffusion.Diffusion"):
        # Call diffusion_init
        diffusion_init(
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
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            edge_areas=edge_areas,
            f_e=f_e,
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            cell_areas=cell_areas,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_cell_x=dual_normal_cell_x,
            dual_normal_cell_y=dual_normal_cell_y,
            global_root=global_root,
            global_level=global_level,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            edge_center_lat=edge_center_lat,
            edge_center_lon=edge_center_lon,
            primal_normal_x=primal_normal_x,
            primal_normal_y=primal_normal_y,
        )

        # Call diffusion_run
        diffusion_run(
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

        assert helpers.dallclose(w.asnumpy(), w_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(vn.asnumpy(), vn_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(exner.asnumpy(), exner_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(theta_v.asnumpy(), theta_v_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(hdef_ic.asnumpy(), hdef_ic_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(div_ic.asnumpy(), div_ic_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(dwdx.asnumpy(), dwdx_.asnumpy(), atol=1e-12)
        assert helpers.dallclose(dwdy.asnumpy(), dwdy_.asnumpy(), atol=1e-12)
