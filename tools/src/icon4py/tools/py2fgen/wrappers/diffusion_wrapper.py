# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

"""
Wrapper module for diffusion granule.

Module contains a diffusion_init and diffusion_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""
import cProfile
import pstats

import gt4py.next as gtx

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    TurbulenceShearForcingType,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import icon
from icon4py.model.common.grid.icon import GlobalGridParams
from icon4py.model.common.grid.vertical import VerticalGrid, VerticalGridConfig
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.tools.common.logger import setup_logger
from icon4py.tools.py2fgen.settings import backend, config as config_settings, device
from icon4py.tools.py2fgen.wrappers import common as wrapper_common
from icon4py.tools.py2fgen.wrappers.debug_utils import print_grid_decomp_info
from icon4py.tools.py2fgen.wrappers.wrapper_dimension import (
    CellGlobalIndexDim,
    CellIndexDim,
    EdgeGlobalIndexDim,
    EdgeIndexDim,
    VertexGlobalIndexDim,
    VertexIndexDim,
)


logger = setup_logger(__name__)

diffusion_wrapper_state = {
    "profiler": cProfile.Profile(),
    "exchange_runtime": definitions.ExchangeRuntime,
}


def profile_enable():
    diffusion_wrapper_state["profiler"].enable()


def profile_disable():
    diffusion_wrapper_state["profiler"].disable()
    stats = pstats.Stats(diffusion_wrapper_state["profiler"])
    stats.dump_stats(f"{__name__}.profile")


def diffusion_init(
    vct_a: gtx.Field[gtx.Dims[dims.KHalfDim], gtx.float64],
    vct_b: gtx.Field[gtx.Dims[dims.KHalfDim], gtx.float64],
    theta_ref_mc: fa.CellKField[wpfloat],
    wgtfac_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KHalfDim], gtx.float64],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    rbf_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.float64],
    rbf_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.float64],
    mask_hdiff: fa.CellKField[bool],
    zd_diffcoef: fa.CellKField[wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.E2CDim, dims.KDim], gtx.int32],
    zd_intcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.E2CDim, dims.KDim], gtx.float64],
    ndyn_substeps: gtx.int32,
    rayleigh_damping_height: gtx.float64,
    nflat_gradp: gtx.int32,
    diffusion_type: gtx.int32,
    hdiff_w: bool,
    hdiff_vn: bool,
    zdiffu_t: bool,
    type_t_diffu: gtx.int32,
    type_vn_diffu: gtx.int32,
    hdiff_efdt_ratio: gtx.float64,
    smagorinski_scaling_factor: gtx.float64,
    hdiff_temp: bool,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    denom_diffu_v: float,
    nudge_max_coeff: float,
    itype_sher: gtx.int32,
    tangent_orientation: fa.EdgeField[wpfloat],
    inverse_primal_edge_lengths: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    edge_areas: fa.EdgeField[wpfloat],
    f_e: fa.EdgeField[wpfloat],
    cell_center_lat: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    cell_center_lon: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    cell_areas: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    dual_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    dual_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    edge_center_lat: fa.EdgeField[wpfloat],
    edge_center_lon: fa.EdgeField[wpfloat],
    primal_normal_x: fa.EdgeField[wpfloat],
    primal_normal_y: fa.EdgeField[wpfloat],
    global_root: gtx.int32,
    global_level: gtx.int32,
    lowest_layer_thickness: gtx.float64,
    model_top_height: gtx.float64,
    stretch_factor: gtx.float64,
):
    logger.info(f"Using Device = {device}")

    global_grid_params = GlobalGridParams(root=global_root, level=global_level)

    if not isinstance(diffusion_wrapper_state["grid"], icon.IconGrid):
        raise Exception(
            "Need to initialise grid using grid_init_diffusion before running diffusion_init."
        )

    # Edge geometry
    edge_params = grid_states.EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=data_alloc.as_1D_sparse_field(primal_normal_vert_x, dims.ECVDim),
        primal_normal_vert_y=data_alloc.as_1D_sparse_field(primal_normal_vert_y, dims.ECVDim),
        dual_normal_vert_x=data_alloc.as_1D_sparse_field(dual_normal_vert_x, dims.ECVDim),
        dual_normal_vert_y=data_alloc.as_1D_sparse_field(dual_normal_vert_y, dims.ECVDim),
        primal_normal_cell_x=data_alloc.as_1D_sparse_field(primal_normal_cell_x, dims.ECDim),
        primal_normal_cell_y=data_alloc.as_1D_sparse_field(primal_normal_cell_y, dims.ECDim),
        dual_normal_cell_x=data_alloc.as_1D_sparse_field(dual_normal_cell_x, dims.ECDim),
        dual_normal_cell_y=data_alloc.as_1D_sparse_field(dual_normal_cell_y, dims.ECDim),
        edge_areas=edge_areas,
        f_e=f_e,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
    )

    # Cell geometry
    cell_params = grid_states.CellParams.from_global_num_cells(
        cell_center_lat=cell_center_lat,
        cell_center_lon=cell_center_lon,
        area=cell_areas,
        global_num_cells=global_grid_params.num_cells,
        length_rescale_factor=1.0,
    )

    # Diffusion parameters
    config = DiffusionConfig(
        diffusion_type=diffusion_type,
        hdiff_w=hdiff_w,
        hdiff_vn=hdiff_vn,
        zdiffu_t=zdiffu_t,
        type_t_diffu=type_t_diffu,
        type_vn_diffu=type_vn_diffu,
        hdiff_efdt_ratio=hdiff_efdt_ratio,
        smagorinski_scaling_factor=smagorinski_scaling_factor,
        hdiff_temp=hdiff_temp,
        n_substeps=ndyn_substeps,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        velocity_boundary_diffusion_denom=denom_diffu_v,
        max_nudging_coeff=nudge_max_coeff / DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO,
        shear_type=TurbulenceShearForcingType(itype_sher),
    )

    diffusion_params = DiffusionParams(config)

    # Vertical grid config
    vertical_config = VerticalGridConfig(
        num_levels=diffusion_wrapper_state["grid"].num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=rayleigh_damping_height,
    )

    # Vertical parameters
    vertical_params = VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
        _min_index_flat_horizontal_grad_pressure=nflat_gradp,
    )

    # Metric state
    metric_state = DiffusionMetricState(
        mask_hdiff=mask_hdiff,
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        zd_intcoef=data_alloc.flatten_first_two_dims(dims.CECDim, dims.KDim, field=zd_intcoef),
        zd_vertoffset=data_alloc.flatten_first_two_dims(
            dims.CECDim, dims.KDim, field=zd_vertoffset
        ),
        zd_diffcoef=zd_diffcoef,
    )

    # Interpolation state
    interpolation_state = DiffusionInterpolationState(
        e_bln_c_s=data_alloc.as_1D_sparse_field(e_bln_c_s, dims.CEDim),
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=data_alloc.as_1D_sparse_field(geofac_div, dims.CEDim),
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    # Initialize the diffusion granule
    diffusion_wrapper_state["granule"] = Diffusion(
        grid=diffusion_wrapper_state["grid"],
        config=config,
        params=diffusion_params,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
        exchange=diffusion_wrapper_state["exchange_runtime"],
    )


def diffusion_run(
    w: gtx.Field[gtx.Dims[dims.CellDim, dims.KHalfDim], gtx.float64],
    vn: fa.EdgeKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    hdef_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KHalfDim], gtx.float64],
    div_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KHalfDim], gtx.float64],
    dwdx: gtx.Field[gtx.Dims[dims.CellDim, dims.KHalfDim], gtx.float64],
    dwdy: gtx.Field[gtx.Dims[dims.CellDim, dims.KHalfDim], gtx.float64],
    dtime: gtx.float64,
    linit: bool,
):
    # prognostic and diagnostic variables
    prognostic_state = PrognosticState(
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
    )

    diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
    )

    if linit:
        diffusion_wrapper_state["granule"].initial_run(
            diagnostic_state,
            prognostic_state,
            dtime,
        )
    else:
        diffusion_wrapper_state["granule"].run(
            prognostic_state=prognostic_state, diagnostic_state=diagnostic_state, dtime=dtime
        )


def grid_init_diffusion(
    cell_starts: gtx.Field[gtx.Dims[CellIndexDim], gtx.int32],
    cell_ends: gtx.Field[gtx.Dims[CellIndexDim], gtx.int32],
    vertex_starts: gtx.Field[gtx.Dims[VertexIndexDim], gtx.int32],
    vertex_ends: gtx.Field[gtx.Dims[VertexIndexDim], gtx.int32],
    edge_starts: gtx.Field[gtx.Dims[EdgeIndexDim], gtx.int32],
    edge_ends: gtx.Field[gtx.Dims[EdgeIndexDim], gtx.int32],
    c2e: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    e2c: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.int32],
    c2e2c: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.int32],
    e2c2e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], gtx.int32],
    e2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2VDim], gtx.int32],
    v2e: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.int32],
    v2c: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], gtx.int32],
    e2c2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.int32],
    c2v: gtx.Field[gtx.Dims[dims.CellDim, dims.C2VDim], gtx.int32],
    c_owner_mask: gtx.Field[[dims.CellDim], bool],
    e_owner_mask: gtx.Field[[dims.EdgeDim], bool],
    v_owner_mask: gtx.Field[[dims.VertexDim], bool],
    c_glb_index: gtx.Field[[CellGlobalIndexDim], gtx.int32],
    e_glb_index: gtx.Field[[EdgeGlobalIndexDim], gtx.int32],
    v_glb_index: gtx.Field[[VertexGlobalIndexDim], gtx.int32],
    comm_id: gtx.int32,
    global_root: gtx.int32,
    global_level: gtx.int32,
    num_vertices: gtx.int32,
    num_cells: gtx.int32,
    num_edges: gtx.int32,
    vertical_size: gtx.int32,
    limited_area: bool,
):
    # todo: write this logic into template.py
    if isinstance(limited_area, int):
        limited_area = bool(limited_area)

    global_grid_params = GlobalGridParams(level=global_level, root=global_root)

    diffusion_wrapper_state["grid"] = wrapper_common.construct_icon_grid(
        cell_starts=cell_starts.ndarray,
        cell_ends=cell_ends.ndarray,
        vertex_starts=vertex_starts.ndarray,
        vertex_ends=vertex_ends.ndarray,
        edge_starts=edge_starts.ndarray,
        edge_ends=edge_ends.ndarray,
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
        global_grid_params=global_grid_params,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=True if config_settings.device == "GPU" else False,
    )

    if config_settings.parallel_run:
        # Set MultiNodeExchange as exchange runtime
        (
            processor_props,
            decomposition_info,
            exchange_runtime,
        ) = wrapper_common.construct_decomposition(
            c_glb_index,
            e_glb_index,
            v_glb_index,
            c_owner_mask,
            e_owner_mask,
            v_owner_mask,
            num_cells,
            num_edges,
            num_vertices,
            vertical_size,
            comm_id,
        )
        print_grid_decomp_info(
            diffusion_wrapper_state["grid"],
            processor_props,
            decomposition_info,
            num_cells,
            num_edges,
            num_vertices,
        )
        # set exchange runtime to MultiNodeExchange
        diffusion_wrapper_state["exchange_runtime"] = exchange_runtime
    else:
        # set exchange runtime to SingleNodeExchange
        diffusion_wrapper_state["exchange_runtime"] = definitions.SingleNodeExchange()
