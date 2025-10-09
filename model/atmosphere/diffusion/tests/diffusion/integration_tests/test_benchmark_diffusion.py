# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import pytest


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing
import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common.constants import RayleighType
from icon4py.model.common.grid import (
    base,
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    vertical as v_grid,
)
from icon4py.model.common.initialization.jablonowski_williamson_topography import (
    jablonowski_williamson_topography,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, grid_utils
from icon4py.model.testing.grid_utils import construct_decomposition_info

from ..fixtures import *  # noqa: F403


@pytest.mark.embedded_remap_error
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "grid", [definitions.Grids.MCH_OPR_R04B07_DOMAIN01, definitions.Grids.R02B07_GLOBAL]
)
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
def test_run_diffusion_benchmark(
    grid: definitions.GridDescription,
    backend: gtx_typing.Backend | None,
    benchmark: Any,
) -> None:
    dtime = 10.0

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

    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid, num_levels=85, keep_skip_values=True, backend=backend
    )

    mesh = grid_manager.grid
    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.geometry_fields

    decomposition_info = construct_decomposition_info(mesh, backend)

    geometry_field_source = grid_geometry.GridGeometry.with_geometry_type(
        grid=mesh,
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
        cell_lat=cell_geometry.cell_center_lat.ndarray,
        u0=35.0,
        backend=backend,
    )

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=mesh,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=mesh,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c),
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=RayleighType.KLEMP,
        rayleigh_coeff=5.0,
        exner_expol=0.333,
        vwind_offctr=0.2,
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
        hdef_ic=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
        div_ic=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
        dwdx=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
        dwdy=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
    )

    prognostic_state = prognostics.PrognosticState(
        w=data_alloc.random_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, low=0.0, allocator=backend
        ),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, allocator=backend),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=backend),
    )

    diffusion_granule = diffusion.Diffusion(
        grid=mesh,
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

    benchmark(diffusion_granule.run, diagnostic_state, prognostic_state, dtime)
