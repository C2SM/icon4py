# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
v2 diffusion bindings: minimal Fortran interface, factory-derived fields.

Unlike v1, ICON passes only primary inputs (grid topology, raw geometry, vertical
coordinate, topography). The interpolation and metric fields are derived by the icon4py
factories. RBF coefficients and mean_cell_area are still passed from Fortran and injected
into the factories (see factory_setup).
"""

import dataclasses
import logging
from collections.abc import Callable

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.grid.states as grid_states
from icon4py.bindings import (
    common as wrapper_common,
    config as wrapper_config,
    icon4py_export,
)
from icon4py.bindings.grid_wrapper import NumpyBoolArray1D, NumpyInt32Array1D
from icon4py.bindings.v2 import diffusion_setup, factory_setup
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    DiffusionType,
    ForcingType,
    SmagorinskyStencilType,
    TemperatureDiscretizationType,
    TurbulenceShearForcingType,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.type_alias import wpfloat


logger = logging.getLogger(__name__)

VertexField = gtx.Field[gtx.Dims[dims.VertexDim], gtx.float64]


@dataclasses.dataclass
class GridStateV2:
    grid: icon_grid.IconGrid
    vertical_grid: v_grid.VerticalGrid
    sources: factory_setup.StaticFieldSources
    edge_geometry: grid_states.EdgeParams
    cell_geometry: grid_states.CellParams
    exchange_runtime: decomposition_defs.ExchangeRuntime
    backend: object
    allocator: object


@dataclasses.dataclass
class DiffusionGranuleV2:
    diffusion: Diffusion
    dummy_field_factory: Callable


grid_state: GridStateV2 | None = None
granule: DiffusionGranuleV2 | None = None


@icon4py_export.export
def grid_init_v2(  # noqa: PLR0917 [too-many-positional-arguments]
    cell_starts: NumpyInt32Array1D,
    cell_ends: NumpyInt32Array1D,
    vertex_starts: NumpyInt32Array1D,
    vertex_ends: NumpyInt32Array1D,
    edge_starts: NumpyInt32Array1D,
    edge_ends: NumpyInt32Array1D,
    c2e: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    e2c: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.int32],
    c2e2c: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.int32],
    e2c2e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], gtx.int32],
    e2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2VDim], gtx.int32],
    v2e: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.int32],
    v2c: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], gtx.int32],
    e2c2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.int32],
    c2v: gtx.Field[gtx.Dims[dims.CellDim, dims.C2VDim], gtx.int32],
    c_owner_mask: NumpyBoolArray1D,
    e_owner_mask: NumpyBoolArray1D,
    v_owner_mask: NumpyBoolArray1D,
    c_glb_index: NumpyInt32Array1D,
    e_glb_index: NumpyInt32Array1D,
    v_glb_index: NumpyInt32Array1D,
    cell_refin_ctrl: NumpyInt32Array1D,
    edge_refin_ctrl: NumpyInt32Array1D,
    vertex_refin_ctrl: NumpyInt32Array1D,
    # raw geometry (replaces the v1 inverse/normal forms; factories derive the rest)
    edge_length: fa.EdgeField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    edge_cell_distance: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    edge_vertex_distance: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2VDim], gtx.float64],
    cell_area: fa.CellField[wpfloat],
    dual_area: VertexField,
    tangent_orientation: fa.EdgeField[wpfloat],
    cell_normal_orientation: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    edge_orientation_on_vertex: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.float64],
    cell_lat: fa.CellField[wpfloat],
    cell_lon: fa.CellField[wpfloat],
    edge_lat: fa.EdgeField[wpfloat],
    edge_lon: fa.EdgeField[wpfloat],
    vertex_lat: VertexField,
    vertex_lon: VertexField,
    vct_a: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    vct_b: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    topography: fa.CellField[wpfloat],
    # injected from Fortran (rounding-sensitive / global-reduction)
    rbf_vec_coeff_v: wrapper_common.Float64Array3D,
    mean_cell_area: gtx.float64,
    nudge_max_coeff: gtx.float64,  # scaled ICON value (not the raw namelist value)
    lowest_layer_thickness: gtx.float64,
    model_top_height: gtx.float64,
    stretch_factor: gtx.float64,
    flat_height: gtx.float64,
    rayleigh_damping_height: gtx.float64,
    comm_id: gtx.int32,
    num_vertices: gtx.int32,
    num_cells: gtx.int32,
    num_edges: gtx.int32,
    vertical_size: gtx.int32,
    limited_area: bool,
    backend: gtx.int32,
) -> None:
    on_gpu = c2e.array_ns != np  # TODO(havogt): expose `on_gpu` from py2fgen
    actual_backend = wrapper_common.select_backend(
        wrapper_common.BackendIntEnum(backend), on_gpu=on_gpu
    )
    allocator = model_backends.get_allocator(actual_backend)
    # The factories need a concrete gt4py Backend (they read `.name`), but select_backend
    # returns a BackendDescriptor dict; resolve it via its backend_factory (gtfn by default).
    if model_backends.is_backend_descriptor(actual_backend):
        backend_factory = actual_backend.get(
            "backend_factory", model_backends.make_custom_gtfn_backend
        )
        resolved_backend = backend_factory(actual_backend["device"])
    else:
        resolved_backend = actual_backend
    xp = cell_area.array_ns

    if comm_id is None:
        decomposition_info = factory_setup.single_node_decomposition_info(
            num_cells=num_cells, num_edges=num_edges, num_vertices=num_vertices
        )
        exchange = decomposition_defs.single_node_exchange
        reductions = decomposition_defs.single_node_reductions
        distributed = False
    else:
        process_props, decomposition_info, exchange = wrapper_common.construct_decomposition(
            c_glb_index=c_glb_index,
            e_glb_index=e_glb_index,
            v_glb_index=v_glb_index,
            c_owner_mask=c_owner_mask,
            e_owner_mask=e_owner_mask,
            v_owner_mask=v_owner_mask,
            num_cells=num_cells,
            num_edges=num_edges,
            num_vertices=num_vertices,
            comm_id=comm_id,
        )
        reductions = decomposition_defs.create_reduction(process_props, decomposition_info)
        distributed = not process_props.is_single_rank()

    grid = wrapper_common.construct_icon_grid(
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e.ndarray,
        e2c=e2c.ndarray,
        c2e2c=c2e2c.ndarray,
        e2c2e=e2c2e.ndarray,
        e2v=e2v.ndarray,
        v2e=v2e.ndarray,
        v2c=v2c.ndarray,
        e2c2v=e2c2v.ndarray,
        c2v=c2v.ndarray,
        grid_id="icon_grid",
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        distributed=distributed,
        allocator=allocator,
        cell_refin_ctrl=cell_refin_ctrl,
        edge_refin_ctrl=edge_refin_ctrl,
        vertex_refin_ctrl=vertex_refin_ctrl,
    )

    vertical_config = v_grid.VerticalGridConfig(
        num_levels=vertical_size,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=rayleigh_damping_height,
        flat_height=flat_height,
    )
    vertical_grid = v_grid.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=vct_b)

    coordinates = factory_setup.build_coordinates(
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
    )
    extra_fields = factory_setup.build_extra_fields(
        edge_length=edge_length,
        dual_edge_length=dual_edge_length,
        edge_cell_distance=edge_cell_distance,
        edge_vertex_distance=edge_vertex_distance,
        cell_area=cell_area,
        dual_area=dual_area,
        tangent_orientation=tangent_orientation,
        cell_normal_orientation=cell_normal_orientation,
        edge_orientation_on_vertex=edge_orientation_on_vertex,
    )

    # rbf_vec_coeff_v arrives in ICON layout (rbf_vec_dim_v, 2, n_vertices); split components.
    rbf_v1 = gtx.as_field(
        [dims.VertexDim, dims.V2EDim], xp.transpose(rbf_vec_coeff_v[:, 0, :]), allocator=allocator
    )
    rbf_v2 = gtx.as_field(
        [dims.VertexDim, dims.V2EDim], xp.transpose(rbf_vec_coeff_v[:, 1, :]), allocator=allocator
    )

    interpolation_config = interpolation_factory.InterpolationConfig(
        max_nudging_coefficient=nudge_max_coeff
    )
    metrics_config = metrics_factory.MetricsConfig()

    sources = factory_setup.build_static_field_sources(
        grid=grid,
        decomposition_info=decomposition_info,
        coordinates=coordinates,
        extra_fields=extra_fields,
        vertical_grid=vertical_grid,
        topography=topography,
        interpolation_config=interpolation_config,
        metrics_config=metrics_config,
        rbf_vec_coeff_v1=rbf_v1,
        rbf_vec_coeff_v2=rbf_v2,
        mean_cell_area=float(mean_cell_area),
        backend=resolved_backend,
        exchange=exchange,
        reductions=reductions,
    )

    global grid_state  # noqa: PLW0603 [global-statement]
    grid_state = GridStateV2(
        grid=grid,
        vertical_grid=vertical_grid,
        sources=sources,
        edge_geometry=diffusion_setup.assemble_edge_params(sources.geometry),
        cell_geometry=diffusion_setup.assemble_cell_params(sources.geometry),
        exchange_runtime=exchange,
        backend=resolved_backend,
        allocator=allocator,
    )


@icon4py_export.export
def diffusion_init_v2(  # noqa: PLR0917 [too-many-positional-arguments]
    ndyn_substeps: gtx.int32,
    diffusion_type: gtx.int32,
    hdiff_w: bool,
    hdiff_vn: bool,
    hdiff_smag_w: bool,
    zdiffu_t: bool,
    type_t_diffu: gtx.int32,
    type_vn_diffu: gtx.int32,
    hdiff_efdt_ratio: gtx.float64,
    hdiff_w_efdt_ratio: gtx.float64,
    smagorinski_scaling_factor: gtx.float64,
    smagorinski_scaling_factor2: gtx.float64,
    smagorinski_scaling_factor3: gtx.float64,
    smagorinski_scaling_factor4: gtx.float64,
    smagorinski_scaling_height: gtx.float64,
    smagorinski_scaling_height2: gtx.float64,
    smagorinski_scaling_height3: gtx.float64,
    smagorinski_scaling_height4: gtx.float64,
    hdiff_temp: bool,
    denom_diffu_v: float,
    nudge_max_coeff: float,  # scaled ICON value (not the namelist value)
    itype_sher: gtx.int32,
    iforcing: gtx.int32,
    a_hshr: gtx.float64,
    loutshs: bool,
) -> None:
    if grid_state is None:
        raise RuntimeError("Need to initialise grid using 'grid_init_v2' before 'diffusion_init_v2'.")

    config = DiffusionConfig(
        diffusion_type=DiffusionType(diffusion_type),
        apply_to_vertical_wind=hdiff_w,
        apply_to_horizontal_wind=hdiff_vn,
        apply_smag_diff_to_vertical_wind=hdiff_smag_w,
        apply_zdiffusion_t=zdiffu_t,
        type_t_diffu=TemperatureDiscretizationType(type_t_diffu),
        type_vn_diffu=SmagorinskyStencilType(type_vn_diffu),
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        hdiff_w_efdt_ratio=hdiff_w_efdt_ratio,
        smagorinski_scaling_factor=smagorinski_scaling_factor,
        smagorinski_scaling_factor2=smagorinski_scaling_factor2,
        smagorinski_scaling_factor3=smagorinski_scaling_factor3,
        smagorinski_scaling_factor4=smagorinski_scaling_factor4,
        smagorinski_scaling_height=smagorinski_scaling_height,
        smagorinski_scaling_height2=smagorinski_scaling_height2,
        smagorinski_scaling_height3=smagorinski_scaling_height3,
        smagorinski_scaling_height4=smagorinski_scaling_height4,
        apply_to_temperature=hdiff_temp,
        ndyn_substeps=int(ndyn_substeps),
        velocity_boundary_diffusion_denominator=denom_diffu_v,
        max_nudging_coefficient=nudge_max_coeff,
        shear_type=TurbulenceShearForcingType(itype_sher),
        iforcing=ForcingType(iforcing),
        a_hshr=a_hshr,
        loutshs=loutshs,
    )
    diffusion_params = DiffusionParams(config)

    interpolation_state, metric_state = diffusion_setup.assemble_diffusion_states(
        grid_state.sources
    )

    global granule  # noqa: PLW0603 [global-statement]
    granule = DiffusionGranuleV2(
        diffusion=Diffusion(
            grid=grid_state.grid,
            config=config,
            params=diffusion_params,
            vertical_grid=grid_state.vertical_grid,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=grid_state.edge_geometry,
            cell_params=grid_state.cell_geometry,
            backend=grid_state.backend,
            exchange=grid_state.exchange_runtime,
        ),
        dummy_field_factory=wrapper_common.cached_dummy_field_factory(grid_state.allocator),
    )
    if wrapper_config.WAIT_FOR_COMPILATION:
        gtx.wait_for_compilation()


@icon4py_export.export
def diffusion_run_v2(  # noqa: PLR0917 [too-many-positional-arguments]
    w: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn: fa.EdgeKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    hdef_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    div_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    dwdx: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    dwdy: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    dtime: gtx.float64,
    linit: bool,
) -> None:
    from icon4py.model.atmosphere.diffusion.diffusion_states import (
        DiffusionDiagnosticState,
    )

    if granule is None:
        raise RuntimeError("Diffusion granule not initialized. Call 'diffusion_init_v2' first.")

    prognostic_state = PrognosticState(w=w, vn=vn, exner=exner, theta_v=theta_v, rho=rho)

    if hdef_ic is None:
        hdef_ic = granule.dummy_field_factory("hdef_ic", domain=w.domain, dtype=w.dtype)
    if div_ic is None:
        div_ic = granule.dummy_field_factory("div_ic", domain=w.domain, dtype=w.dtype)
    if dwdx is None:
        dwdx = granule.dummy_field_factory("dwdx", domain=w.domain, dtype=w.dtype)
    if dwdy is None:
        dwdy = granule.dummy_field_factory("dwdy", domain=w.domain, dtype=w.dtype)
    diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=hdef_ic, div_ic=div_ic, dwdx=dwdx, dwdy=dwdy
    )

    granule.diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
        initial_run=linit,
    )
