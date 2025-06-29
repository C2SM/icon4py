# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools

import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states, diffusion_utils
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common.utils.data_allocation import zero_field
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils,
    helpers,
    reference_funcs as ref_funcs,
    serialbox as sb,
)

from icon4py.model.common.interpolation import (
    interpolation_attributes,
    interpolation_factory
)
from icon4py.model.common.metrics import (
    metrics_attributes,
    metrics_factory,
)
from icon4py.model.testing.datatest_fixtures import grid_savepoint

from .utils import (
    compare_dace_orchestration_multiple_steps,
    construct_diffusion_config,
    diff_multfac_vn_numpy,
    smag_limit_numpy,
    verify_diffusion_fields,
)


grid_functionality = {dt_utils.GLOBAL_EXPERIMENT: {}, dt_utils.REGIONAL_EXPERIMENT: {}}


def get_grid_for_experiment(experiment, backend):
    return _get_or_initialize(experiment, backend, "grid")


def get_edge_geometry_for_experiment(experiment, backend):
    return _get_or_initialize(experiment, backend, "edge_geometry")


def get_cell_geometry_for_experiment(experiment, backend):
    return _get_or_initialize(experiment, backend, "cell_geometry")


def _get_or_initialize(experiment, backend, name):
    grid_file = (
        dt_utils.REGIONAL_EXPERIMENT
        if experiment == dt_utils.REGIONAL_EXPERIMENT
        else dt_utils.R02B04_GLOBAL
    )

    if not grid_functionality[experiment].get(name):
        geometry_ = grid_utils.get_grid_geometry(backend, grid_file, num_levels)
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
            primal_normal_vert_x=data_alloc.flatten_first_two_dims(
                dims.ECVDim,
                field=(geometry_.get(geometry_meta.EDGE_NORMAL_VERTEX_U)),
                backend=backend,
            ),
            primal_normal_vert_y=data_alloc.flatten_first_two_dims(
                dims.ECVDim,
                field=(geometry_.get(geometry_meta.EDGE_NORMAL_VERTEX_V)),
                backend=backend,
            ),
            dual_normal_cell_x=geometry_.get(geometry_meta.EDGE_TANGENT_CELL_U),
            dual_normal_cell_y=geometry_.get(geometry_meta.EDGE_TANGENT_CELL_V),
            dual_normal_vert_x=data_alloc.flatten_first_two_dims(
                dims.ECVDim,
                field=geometry_.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
                backend=backend,
            ),
            dual_normal_vert_y=data_alloc.flatten_first_two_dims(
                dims.ECVDim,
                field=geometry_.get(geometry_meta.EDGE_TANGENT_VERTEX_V),
                backend=backend,
            ),
        )
        grid_functionality[experiment]["grid"] = grid
        grid_functionality[experiment]["edge_geometry"] = edge_params
        grid_functionality[experiment]["cell_geometry"] = cell_params
    return grid_functionality[experiment].get(name)

def metrics_config(experiment: str) -> tuple:
    rayleigh_coeff = 5.0
    lowest_layer_thickness = 50.0
    exner_expol = 0.333
    vwind_offctr = 0.2
    rayleigh_type = 2
    model_top_height = 23500.0
    stretch_factor = 1.0
    damping_height = 45000.0
    match experiment:
        case dt_utils.REGIONAL_EXPERIMENT:
            lowest_layer_thickness = 20.0
            model_top_height = 23000.0
            stretch_factor = 0.65
            damping_height = 12500.0
        case dt_utils.GLOBAL_EXPERIMENT:
            model_top_height = 75000.0
            stretch_factor = 0.9
            damping_height = 50000.0
            rayleigh_coeff = 0.1
            exner_expol = 0.3333333333333
            vwind_offctr = 0.15

    return (
        lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type,
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
    assert all(p >= 0 for p in params.smagorinski_factor)


@pytest.mark.infinite_concat_where
@pytest.mark.datatest
def test_diffusion_init(
    savepoint_diffusion_init,
    interpolation_savepoint,
    metrics_savepoint,
    experiment,
    step_date_init,
    interface_model_height,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
):
    # TODO: grid file can be put as input?
    grid_file = (
        dt_utils.REGIONAL_EXPERIMENT
        if experiment == dt_utils.REGIONAL_EXPERIMENT
        else dt_utils.R02B04_GLOBAL
    )
    config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    (lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type) = metrics_config(experiment)

    grid = get_grid_for_experiment(experiment, backend)
    cell_params = get_cell_geometry_for_experiment(experiment, backend)
    edge_params = get_edge_geometry_for_experiment(experiment, backend)

    geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)

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

    meta = savepoint_diffusion_init.get_metadata("linit", "date")

    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=geometry._decomposition_info,
        geometry_source=geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid= grid,
        vertical_grid=vertical_config,
        decomposition_info=geometry._decomposition_info,
        geometry_source = geometry,
        interpolation_source = interpolation_field_source,
        backend = backend,
        metadata = metrics_attributes.attrs,
        interface_model_height = interface_model_height,
        e_refin_ctrl=grid_savepoint.refin_ctrl(dims.EdgeDim), # TODO (Yilu): refin_ctrl should be moved from refin_ctrl
        c_refin_ctrl=grid_savepoint.refin_ctrl(dims.CellDim), # TODO:
        damping_height=damping_height,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
    )

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s = interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        # TODO (Yilu) : wait for rbf
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        # TODO (Yilu):
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )

    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_field_source.get(metrics_attributes.MASK_HDIFF),
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF_DSL),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET_DSL),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF_DSL),
    )

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
    )

    assert diffusion_granule.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float
    )

    assert helpers.dallclose(diffusion_granule.v_vert.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.u_vert.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.kh_smag_ec.asnumpy(), 0.0)
    assert helpers.dallclose(diffusion_granule.kh_smag_e.asnumpy(), 0.0)

    shape_k = (grid.num_levels,)
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
        vertical_params.vct_a.ndarray,
    )
    assert helpers.dallclose(diffusion_granule.enh_smag_fac.asnumpy(), expected_enh_smag_fac)


def _verify_init_values_against_savepoint(
    savepoint: sb.IconDiffusionInitSavepoint, diffusion_granule: diffusion.Diffusion, backend
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


@pytest.mark.infinite_concat_where
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
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_diffusion_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
    interface_model_height,
):
    # TODO: or the grid file can be passed
    grid_file = (
        dt_utils.REGIONAL_EXPERIMENT
        if experiment == dt_utils.REGIONAL_EXPERIMENT
        else dt_utils.R02B04_GLOBAL
    )
    (lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type) = metrics_config(experiment)
    grid = get_grid_for_experiment(experiment, backend)
    cell_params = get_cell_geometry_for_experiment(experiment, backend)
    edge_params = get_edge_geometry_for_experiment(experiment, backend)
    geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)
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

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid = grid,
        decomposition_info=geometry._decomposition_info,
        geometry_source = geometry,
        backend = backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid= grid,
        vertical_grid=vertical_config,
        decomposition_info=geometry._decomposition_info,
        geometry_source = geometry,
        interpolation_source = interpolation_field_source,
        backend = backend,
        metadata = metrics_attributes.attrs,
        interface_model_height = interface_model_height,
        e_refin_ctrl=grid_savepoint.refin_ctrl(dims.EdgeDim), # TODO (Yilu): refin_ctrl should be moved from refin_ctrl
        c_refin_ctrl=grid_savepoint.refin_ctrl(dims.CellDim), # TODO:
        damping_height=damping_height,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
    )

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s = interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        # TODO (Yilu) : wait for rbf
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        # TODO (Yilu):
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )

    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_field_source.get(metrics_attributes.MASK_HDIFF),
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF_DSL),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET_DSL),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF_DSL),
    )

    diffusion_granule = diffusion.Diffusion(
        grid,
        config,
        additional_parameters,
        vertical_params,
        metric_state,
        interpolation_state,
        edge_params,
        cell_params,
        orchestration=False,
        backend=backend,
    )

    _verify_init_values_against_savepoint(savepoint_diffusion_init, diffusion_granule, backend)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "grid_file",
    [   # TODO: ingnore regional
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", [2]) # TODO: the default value is 5
def test_run_diffusion_single_step(
    grid_file,
    interface_model_height,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
    orchestration,
    benchmark,
):
    # TODO: focus on the global, pass the grid_file

    grid_file = (
        dt_utils.REGIONAL_EXPERIMENT
        if experiment == dt_utils.REGIONAL_EXPERIMENT
        else dt_utils.R02B04_GLOBAL
    )

    (lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type) = metrics_config(experiment)

    grid = get_grid_for_experiment(experiment, backend)
    cell_geometry = get_cell_geometry_for_experiment(experiment, backend)
    edge_geometry = get_edge_geometry_for_experiment(experiment, backend)
    geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)

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

    #TODO: manually set the config
    config = construct_diffusion_config(experiment, ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    # TODO (yilu): set as default value
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=geometry._decomposition_info,
        geometry_source=geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=grid,
        vertical_grid=vertical_config,
        decomposition_info=geometry._decomposition_info,
        geometry_source=geometry,
        topography= , # TODO: add topography as the input to the metrics field factory
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        interface_model_height=interface_model_height,
        e_refin_ctrl=grid_savepoint.refin_ctrl(dims.EdgeDim), # TODO: this goes to grid, should be removed from metrics_factory
        c_refin_ctrl=grid_savepoint.refin_ctrl(dims.CellDim),  # TODO (Yilu): refin_ctrl should be moved from refin_ctrl
        damping_height=damping_height,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
    )

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s = interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        # TODO (Yilu) : wait for rbf (random values for now)
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        # TODO (Yilu): the compute function should also be registered in interpolationfieldfactory
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(), # TODO: compute_nudgeco: programprofield
    )

    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_field_source.get(metrics_attributes.MASK_HDIFF),
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF_DSL),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET_DSL),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF_DSL),
    )
    # TODO: we should compute mannually the diagnostic states
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=zero_field(),
        div_ic=zero_field(),
        dwdx=zero_field(),
        dwdy=zero_field(),
    )
    # TODO: initialize random values
    prognostic_state = savepoint_diffusion_init.construct_prognostics()

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
    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_init)
    # assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    diffusion_granule.run
    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_init)

    helpers.run_verify_and_benchmark(
        functools.partial(
            diffusion_granule.run,
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
        ),
        functools.partial(
            verify_diffusion_fields,
            config=config,
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            diffusion_savepoint=savepoint_diffusion_exit,
        ),
        benchmark, # TODO: in the old, delete benchmark, but keep benchmark as input argument in the new test
    )


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
    experiment,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    backend,
    icon_grid,
):
    if not helpers.is_dace(backend):
        raise pytest.skip("This test is only executed for dace backends")
    if data_alloc.is_cupy_device(backend):
        pytest.xfail("GPU compilation fails.")
    ######################################################################
    # Diffusion initialization
    ######################################################################
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_savepoint.e_bln_c_s(),
            backend=backend,
        ),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_savepoint.geofac_div(),
            backend=backend,
        ),
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

    diagnostic_state_dace_non_orch = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state_dace_non_orch = savepoint_diffusion_init.construct_prognostics()

    diffusion_granule = diffusion.Diffusion(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        orchestration=False,
        backend=backend,
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

    diagnostic_state_dace_orch = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state_dace_orch = savepoint_diffusion_init.construct_prognostics()

    diffusion_granule = diffusion.Diffusion(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        orchestration=True,
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
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize("linit", [True])
@pytest.mark.parametrize("orchestration", [False, True])
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
    backend,
    orchestration,
):
    if orchestration and not helpers.is_dace(backend):
        pytest.skip("Orchestration test requires a dace backend.")
    if orchestration and data_alloc.is_cupy_device(backend):
        pytest.xfail("GPU compilation fails.")
    grid = get_grid_for_experiment(experiment, backend)
    cell_geometry = get_cell_geometry_for_experiment(experiment, backend)
    edge_geometry = get_edge_geometry_for_experiment(experiment, backend)
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")

    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_savepoint.e_bln_c_s(),
            backend=backend,
        ),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_savepoint.geofac_div(),
            backend=backend,
        ),
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
    config = construct_diffusion_config(experiment, ndyn_substeps=2)
    params = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=params,
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        orchestration=orchestration,
    )

    assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

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
