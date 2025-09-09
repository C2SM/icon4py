# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from typing import Any, Dict

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend
from icon4py.model.atmosphere.diffusion import diffusion
import icon4py.model.common.dimension as dims
from icon4py.model.common.grid import geometry as grid_geometry
from icon4py.model.common.grid import geometry_attributes as geometry_meta
from icon4py.model.common.grid import vertical as v_grid
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common.initialization.jablonowski_williamson_topography import (
    jablonowski_williamson_topography,
)
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.stencil_tests import construct_dummy_decomposition_info
from icon4py.model.testing import definitions
from icon4py.model.testing import grid_utils
from ..fixtures import *


@pytest.mark.embedded_remap_error
@pytest.mark.benchmark(
    group="diffusion_benchmark",
)
@pytest.mark.parametrize("grid", [definitions.Grids.MCH_OPR_R19B08_DOMAIN01])
def test_run_diffusion_benchmark(
    grid: definitions.GridDescription,
    vertical_grid_params: Dict[str, float],
    metrics_factory_params: Dict[str, Any],
    backend: Any,
    benchmark: Any,
) -> None:
    dtime = 10.0
    grid_manager = grid_utils.get_grid_manager_from_identifier(grid, num_levels=80, backend=backend)
    config = diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=False,
        thslp_zdiffu=0.02,
        thhgtd_zdiffu=125.0,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coefficient=0.375,
        n_substeps=5,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )

    diffusion_parameters = diffusion.DiffusionParams(config)

    mesh = grid_manager.grid
    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.geometry_fields

    decomposition_info = construct_dummy_decomposition_info(mesh, backend)

    geometry_field_source = grid_geometry.GridGeometry(
        grid=grid,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
    )
    edge_geometry = grid_states.EdgeParams(
        edge_center_lat=geometry_field_source.get(geometry_meta.EDGE_LAT),
        edge_center_lon=geometry_field_source.get(geometry_meta.EDGE_LON),
        tangent_orientation=geometry_field_source.get(geometry_meta.TANGENT_ORIENTATION),
        coriolis_frequency=geometry_field_source.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_areas=geometry_field_source.get(geometry_meta.EDGE_AREA),
        primal_edge_lengths=geometry_field_source.get(geometry_meta.EDGE_LENGTH),
        inverse_primal_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.EDGE_LENGTH}"
        ),
        dual_edge_lengths=geometry_field_source.get(geometry_meta.DUAL_EDGE_LENGTH),
        inverse_dual_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ),
        inverse_vertex_vertex_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
        ),
        primal_normal_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_U),
        primal_normal_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_V),
        primal_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_U),
        primal_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_V),
        primal_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
        primal_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        dual_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_U),
        dual_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_V),
        dual_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
        dual_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
    )

    topo_c = jablonowski_williamson_topography(
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
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
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

    benchmark.pedantic(
        diffusion_granule.run,
        args=(diagnostic_state, prognostic_state, dtime),
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )
