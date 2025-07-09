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
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import (
    metrics_attributes,
    metrics_factory,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils,
    helpers,
)

from .utils import (
    construct_diffusion_config,
    verify_diffusion_fields,
)


# TODO(Yilu): remove the grid file

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
        geometry_ = grid_utils.get_grid_geometry(backend, experiment, grid_file)
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


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "grid_file, experiment, step_date_init, step_date_exit",
    [  # TODO: ingnore regional
        # (
        #     dt_utils.REGIONAL_EXPERIMENT,
        #     dt_utils.REGIONAL_EXPERIMENT,
        #     "2021-06-20T12:00:10.000",
        #     "2021-06-20T12:00:10.000",
        # ),
        (
            dt_utils.GLOBAL_EXPERIMENT,
            dt_utils.GLOBAL_EXPERIMENT,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", [2])  # TODO: the default value is 5
@pytest.mark.parametrize("orchestration", [False])
def test_run_diffusion_single_step(
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    grid_file,
    experiment,
    topography_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    rayleigh_coeff,
    exner_expol,
    vwind_offctr,
    rayleigh_type,
    ndyn_substeps,
    backend,
    orchestration,
    benchmark,
):
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

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

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
        vertical_grid=vertical_grid,
        decomposition_info=geometry._decomposition_info,
        geometry_source=geometry,
        topography=topography_savepoint.topo_c(),
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        e_refin_ctrl=grid.refinement_control[dims.EdgeDim],
        c_refin_ctrl=grid.refinement_control[dims.CellDim],
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
    )

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
            backend=backend,
        ),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
            backend=backend,
        ),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )

    metric_state = diffusion_states.DiffusionMetricState(
        mask_hdiff=metrics_field_source.get(metrics_attributes.MASK_HDIFF),
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF_DSL),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET_DSL),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF_DSL),
    )

    config = construct_diffusion_config(experiment, ndyn_substeps)
    additional_parameters = diffusion.DiffusionParams(config)

    # TODO: we should compute mannually the diagnostic states and prognostic states?
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state = savepoint_diffusion_init.construct_prognostics()
    # TODO:

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        orchestration=orchestration,
    )
    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_init)
    assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

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
        benchmark,
    )
