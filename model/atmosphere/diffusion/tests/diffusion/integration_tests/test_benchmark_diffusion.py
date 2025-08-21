# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools

import pytest
import netCDF4 as nc4
from pathlib import Path
import numpy as np
from icon4py.model.common.initialization.topography_initialization import topography_initialization
import icon4py.model.common.dimension as dims
import gt4py.next as gtx
from icon4py.model.common.decomposition import definitions
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states

from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    vertical as v_grid, geometry,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import (
    metrics_attributes,
    metrics_factory,
)
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils,
)
from icon4py.model.testing.data_handling import download_and_extract

import icon4py.model.common.states.prognostic_state as prognostics

from icon4py.model.common.grid import geometry as grid_geometry

from ..fixtures import *

grid_functionality = {dt_utils.R02B04_GLOBAL: {}, dt_utils.REGIONAL_EXPERIMENT: {}}

def get_edge_geometry_for_grid_file(grid_file, geometry_factory, backend):
    return _get_or_initialize_for_grid_file(grid_file, geometry_factory, backend, "edge_geometry")


def get_cell_geometry_for_grid_file(grid_file, geometry_factory, backend):
    return _get_or_initialize_for_grid_file(grid_file, geometry_factory, backend, "cell_geometry")


def _get_or_initialize_for_grid_file(grid_file, geometry_factory, backend, name):
    """This is refactored from _get_or_initialize in test_diffusion.py,
    which is used to get the edge or cell geometry for a given grid file, not a given experiment"""

    if not grid_functionality[grid_file].get(name):

        cell_params = grid_states.CellParams(
            cell_center_lat=geometry_factory.get(geometry_meta.CELL_LAT),
            cell_center_lon=geometry_factory.get(geometry_meta.CELL_LON),
            area=geometry_factory.get(geometry_meta.CELL_AREA),
        )
        edge_params = grid_states.EdgeParams(
            edge_center_lat=geometry_factory.get(geometry_meta.EDGE_LAT),
            edge_center_lon=geometry_factory.get(geometry_meta.EDGE_LON),
            tangent_orientation=geometry_factory.get(geometry_meta.TANGENT_ORIENTATION),
            coriolis_frequency=geometry_factory.get(geometry_meta.CORIOLIS_PARAMETER),
            edge_areas=geometry_factory.get(geometry_meta.EDGE_AREA),
            primal_edge_lengths=geometry_factory.get(geometry_meta.EDGE_LENGTH),
            inverse_primal_edge_lengths=geometry_factory.get(f"inverse_of_{geometry_meta.EDGE_LENGTH}"),
            dual_edge_lengths=geometry_factory.get(geometry_meta.DUAL_EDGE_LENGTH),
            inverse_dual_edge_lengths=geometry_factory.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"),
            inverse_vertex_vertex_lengths=geometry_factory.get(
                f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
            ),
            primal_normal_x=geometry_factory.get(geometry_meta.EDGE_NORMAL_U),
            primal_normal_y=geometry_factory.get(geometry_meta.EDGE_NORMAL_V),
            primal_normal_cell_x=geometry_factory.get(geometry_meta.EDGE_NORMAL_CELL_U),
            primal_normal_cell_y=geometry_factory.get(geometry_meta.EDGE_NORMAL_CELL_V),
            primal_normal_vert_x=geometry_factory.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
            primal_normal_vert_y=geometry_factory.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
            dual_normal_cell_x=geometry_factory.get(geometry_meta.EDGE_TANGENT_CELL_U),
            dual_normal_cell_y=geometry_factory.get(geometry_meta.EDGE_TANGENT_CELL_V),
            dual_normal_vert_x=geometry_factory.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
            dual_normal_vert_y=geometry_factory.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        )
        grid_functionality[grid_file]["edge_geometry"] = edge_params
        grid_functionality[grid_file]["cell_geometry"] = cell_params
    return grid_functionality[grid_file].get(name)

def construct_dummy_decomposition_info(grid, backend) -> definitions.DecompositionInfo:
    """A public helper function to construct a dummy decomposition info object for test cases
     refactored from grid_utils.py"""

    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)

    def _add_dimension(dim: gtx.Dimension):
        indices = data_alloc.index_field(grid, dim, backend=backend)
        owner_mask = xp.ones((grid.size[dim],), dtype=bool)
        decomposition_info.with_dimension(dim, indices.ndarray, owner_mask)


    decomposition_info = definitions.DecompositionInfo(klevels=grid.num_levels)
    _add_dimension(dims.EdgeDim)
    _add_dimension(dims.VertexDim)
    _add_dimension(dims.CellDim)

    return decomposition_info

@pytest.fixture
def vertical_grid_params(
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
):
    """Group vertical grid configuration parameters into a dictionary."""
    return {
        "lowest_layer_thickness": lowest_layer_thickness,
        "model_top_height": model_top_height,
        "stretch_factor": stretch_factor,
        "damping_height": damping_height,
    }


@pytest.fixture
def metrics_factory_params(
    rayleigh_coeff,
    exner_expol,
    vwind_offctr,
    rayleigh_type,
):
    """Group rayleigh damping configuration parameters into a dictionary."""
    return {
        "rayleigh_coeff": rayleigh_coeff,
        "exner_expol": exner_expol,
        "vwind_offctr": vwind_offctr,
        "rayleigh_type": rayleigh_type,
    }


@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "grid_file",
    [
        (dt_utils.R02B04_GLOBAL),
        #(dt_utils.REGIONAL_GRIDFILE),
    ],
)
@pytest.mark.benchmark(
    group="diffusion_benchmark",
)
def test_run_diffusion_benchmark(
    grid_file,
    vertical_grid_params,
    metrics_factory_params,
    backend,
    benchmark,
):

    itopo = 1 if grid_file == dt_utils.REGIONAL_GRIDFILE else 0

    # get configuration
    num_levels = 65
    dtime = 10.0
    # TODO (Yilu): for now we use the default configuration, later we can add more configurations
    # TODO (Yilu): later we will use the configuration from the grid file
    config=diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=False, # TODO (Yilu): zdiffu_t is the issue
        thslp_zdiffu=0.02,
        thhgtd_zdiffu=125.0,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coefficient=0.375,
        n_substeps=5,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )

    #config = construct_diffusion_config(grid_file, ndyn_substeps)
    diffusion_parameters = diffusion.DiffusionParams(config)

    # run the grid manager to get the grid, coordinates, geometry_fields
    grid_manager = grid_utils.get_grid_manager_from_identifier(grid_file_identifier=grid_file, num_levels=num_levels, keep_skip_values=True,backend=backend)
    grid = grid_manager.grid
    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.geometry_fields

    geometry_field_source = grid_geometry.GridGeometry(
        grid=grid,
        decomposition_info=construct_dummy_decomposition_info(grid, backend),
        backend=backend,
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs
    )

    cell_geometry = get_cell_geometry_for_grid_file(grid_file, geometry_field_source, backend)
    edge_geometry = get_edge_geometry_for_grid_file(grid_file, geometry_field_source, backend)

    if itopo == 1:
        download_and_extract(
            "https://polybox.ethz.ch/index.php/s/CWWtBHBC9iNpLEo/download",
            Path("./extpar_data/"),
            "extpar_r04b09.tar.gz",
        )
        f = nc4.Dataset(
            "/Users/chenyilu/Desktop/EXCLAIM/icon4py/model/atmosphere/diffusion/tests/diffusion_tests/extpar_data/extpar_r04b09.nc",
            "r")
        topo_c = f.variables["topography_c"][:]
        f.close()
    else:
        topo_c =  topography_initialization(
            cell_lat=cell_geometry.cell_center_lat.asnumpy(),
            u0=35.0,
            backend=backend,
        )

    vertical_config = v_grid.VerticalGridConfig(
        grid.num_levels,
        lowest_layer_thickness=vertical_grid_params["lowest_layer_thickness"],
        model_top_height=vertical_grid_params["model_top_height"],
        stretch_factor=vertical_grid_params["stretch_factor"],
        rayleigh_damping_height=vertical_grid_params["damping_height"],
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )


    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=construct_dummy_decomposition_info(grid, backend),
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=grid,
        vertical_grid=vertical_grid,
        decomposition_info=construct_dummy_decomposition_info(grid, backend),
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c),
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=metrics_factory_params["rayleigh_type"],
        rayleigh_coeff=metrics_factory_params["rayleigh_coeff"],
        exner_expol=metrics_factory_params["exner_expol"],
        vwind_offctr=metrics_factory_params["vwind_offctr"],
    )

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
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

    # initialization of the diagnostic and prognostic state
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
        div_ic=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
        dwdx=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
        dwdy=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
    )

    prognostic_state = prognostics.PrognosticState(
        w=data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=0.0),
        vn=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
        exner=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
        theta_v=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
        rho=data_alloc.random_field(grid, dims.CellDim, dims.KDim),
    )

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=diffusion_parameters,
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        orchestration=False,
    )

    benchmark.pedantic(diffusion_granule.run,
        args=(diagnostic_state,prognostic_state,dtime),
        rounds=10, warmup_rounds=2, iterations=1
    )


