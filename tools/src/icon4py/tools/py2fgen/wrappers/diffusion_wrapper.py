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
from typing import Optional

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
from icon4py.model.common.grid import icon
from icon4py.model.common.grid.icon import GlobalGridParams
from icon4py.model.common.grid.vertical import VerticalGrid, VerticalGridConfig
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.tools.common.logger import setup_logger
from icon4py.tools.py2fgen import settings as settings
from icon4py.tools.py2fgen.settings import backend, device
from icon4py.tools.py2fgen.wrappers import grid_wrapper


logger = setup_logger(__name__)

profiler = cProfile.Profile()
granule: Optional[Diffusion] = None


def profile_enable():
    global profiler
    profiler.enable()


def profile_disable():
    global profiler
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(f"{__name__}.profile")


def diffusion_init(
    vct_a: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    vct_b: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    theta_ref_mc: fa.CellKField[wpfloat],
    wgtfac_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    rbf_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.float64],
    rbf_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.float64],
    mask_hdiff: Optional[fa.CellKField[bool]],
    zd_diffcoef: Optional[fa.CellKField[wpfloat]],
    zd_vertoffset: Optional[gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32]],
    zd_intcoef: Optional[gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.float64]],
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
    ltkeshs: bool,
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

    if not isinstance(grid_wrapper.grid, icon.IconGrid):
        raise Exception(
            "Need to initialise grid using grid_init_diffusion before running diffusion_init."
        )

    # Edge geometry
    edge_params = grid_states.EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=data_alloc.flatten_first_two_dims(
            dims.ECVDim, field=primal_normal_vert_x
        ),
        primal_normal_vert_y=data_alloc.flatten_first_two_dims(
            dims.ECVDim, field=primal_normal_vert_y
        ),
        dual_normal_vert_x=data_alloc.flatten_first_two_dims(dims.ECVDim, field=dual_normal_vert_x),
        dual_normal_vert_y=data_alloc.flatten_first_two_dims(dims.ECVDim, field=dual_normal_vert_y),
        primal_normal_cell_x=data_alloc.flatten_first_two_dims(
            dims.ECDim, field=primal_normal_cell_x
        ),
        primal_normal_cell_y=data_alloc.flatten_first_two_dims(
            dims.ECDim, field=primal_normal_cell_y
        ),
        dual_normal_cell_x=data_alloc.flatten_first_two_dims(dims.ECDim, field=dual_normal_cell_x),
        dual_normal_cell_y=data_alloc.flatten_first_two_dims(dims.ECDim, field=dual_normal_cell_y),
        edge_areas=edge_areas,
        coriolis_frequency=f_e,
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
        ltkeshs=ltkeshs,
    )

    diffusion_params = DiffusionParams(config)

    # Vertical grid config
    vertical_config = VerticalGridConfig(
        num_levels=grid_wrapper.grid.num_levels,
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

    nlev = wgtfac_c.domain[dims.KDim].unit_range.stop - 1  # wgtfac_c has nlevp1 levels
    cell_k_domain = {dims.CellDim: wgtfac_c.domain[dims.CellDim].unit_range, dims.KDim: nlev}
    c2e2c_size = geofac_grg_x.domain[dims.C2E2CODim].unit_range.stop - 1
    cell_c2e2c_k_domain = {
        dims.CellDim: wgtfac_c.domain[dims.CellDim].unit_range,
        dims.C2E2CDim: c2e2c_size,
        dims.KDim: nlev,
    }
    xp = wgtfac_c.array_ns
    if mask_hdiff is None:
        mask_hdiff = gtx.zeros(cell_k_domain, dtype=xp.bool_)
    if zd_diffcoef is None:
        zd_diffcoef = gtx.zeros(cell_k_domain, dtype=theta_ref_mc.dtype)
    if zd_intcoef is None:
        zd_intcoef = gtx.zeros(cell_c2e2c_k_domain, dtype=wgtfac_c.dtype)
    if zd_vertoffset is None:
        zd_vertoffset = gtx.zeros(cell_c2e2c_k_domain, dtype=xp.int32)
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
        e_bln_c_s=data_alloc.flatten_first_two_dims(dims.CEDim, field=e_bln_c_s),
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=data_alloc.flatten_first_two_dims(dims.CEDim, field=geofac_div),
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    # Initialize the diffusion granule
    global granule
    granule = Diffusion(
        grid=grid_wrapper.grid,
        config=config,
        params=diffusion_params,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
        exchange=grid_wrapper.exchange_runtime,
    )


def diffusion_run(
    w: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn: fa.EdgeKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    hdef_ic: Optional[gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64]],
    div_ic: Optional[gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64]],
    dwdx: Optional[gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64]],
    dwdy: Optional[gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64]],
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

    if hdef_ic is None:
        hdef_ic = gtx.zeros(w.domain, dtype=w.dtype, allocator=backend)
    if div_ic is None:
        div_ic = gtx.zeros(w.domain, dtype=w.dtype, allocator=backend)
    if dwdx is None:
        dwdx = gtx.zeros(w.domain, dtype=w.dtype, allocator=backend)
    if dwdy is None:
        dwdy = gtx.zeros(w.domain, dtype=w.dtype, allocator=backend)
    diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
    )

    global granule
    if linit:
        granule.initial_run(
            diagnostic_state,
            prognostic_state,
            dtime,
        )
    else:
        granule.run(
            prognostic_state=prognostic_state, diagnostic_state=diagnostic_state, dtime=dtime
        )
