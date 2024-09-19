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

from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import float64, int32
from icon4py.model.atmosphere.diffusion.diffusion import (
    DiffusionConfig,
    DiffusionParams,
    TurbulenceShearForcingType,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO
from icon4py.model.common.grid import geometry
from icon4py.model.common.grid.icon import GlobalGridParams
from icon4py.model.common.grid.vertical import VerticalGrid, VerticalGridConfig
from icon4py.model.common.settings import device
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field
from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.wrappers import common

logger = setup_logger(__name__)


def diffusion_init(
    vct_a: Field[[dims.KHalfDim], float64],
    vct_b: Field[[dims.KHalfDim], float64],
    theta_ref_mc: Field[[dims.CellDim, dims.KDim], float64],
    wgtfac_c: Field[[dims.CellDim, dims.KHalfDim], float64],
    e_bln_c_s: Field[[dims.CellDim, dims.C2EDim], float64],
    geofac_div: Field[[dims.CellDim, dims.C2EDim], float64],
    geofac_grg_x: Field[[dims.CellDim, dims.C2E2CODim], float64],
    geofac_grg_y: Field[[dims.CellDim, dims.C2E2CODim], float64],
    geofac_n2s: Field[[dims.CellDim, dims.C2E2CODim], float64],
    nudgecoeff_e: Field[[dims.EdgeDim], float64],
    rbf_coeff_1: Field[[dims.VertexDim, dims.V2EDim], float64],
    rbf_coeff_2: Field[[dims.VertexDim, dims.V2EDim], float64],
    mask_hdiff: Field[[dims.CellDim, dims.KDim], bool],
    zd_diffcoef: Field[[dims.CellDim, dims.KDim], float64],
    zd_vertoffset: Field[[dims.CellDim, dims.E2CDim, dims.KDim], int32],
    zd_intcoef: Field[[dims.CellDim, dims.E2CDim, dims.KDim], float64],
    ndyn_substeps: int32,
    rayleigh_damping_height: float64,
    nflat_gradp: int32,
    diffusion_type: int32,
    hdiff_w: bool,
    hdiff_vn: bool,
    zdiffu_t: bool,
    type_t_diffu: int32,
    type_vn_diffu: int32,
    hdiff_efdt_ratio: float64,
    smagorinski_scaling_factor: float64,
    hdiff_temp: bool,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    denom_diffu_v: float,
    nudge_max_coeff: float,
    itype_sher: int32,
    tangent_orientation: Field[[dims.EdgeDim], float64],
    inverse_primal_edge_lengths: Field[[dims.EdgeDim], float64],
    inv_dual_edge_length: Field[[dims.EdgeDim], float64],
    inv_vert_vert_length: Field[[dims.EdgeDim], float64],
    edge_areas: Field[[dims.EdgeDim], float64],
    f_e: Field[[dims.EdgeDim], float64],
    cell_center_lat: Field[[dims.CellDim], float64],
    cell_center_lon: Field[[dims.CellDim], float64],
    cell_areas: Field[[dims.CellDim], float64],
    primal_normal_vert_x: Field[[dims.EdgeDim, dims.E2C2VDim], float64],
    primal_normal_vert_y: Field[[dims.EdgeDim, dims.E2C2VDim], float64],
    dual_normal_vert_x: Field[[dims.EdgeDim, dims.E2C2VDim], float64],
    dual_normal_vert_y: Field[[dims.EdgeDim, dims.E2C2VDim], float64],
    primal_normal_cell_x: Field[[dims.EdgeDim, dims.E2CDim], float64],
    primal_normal_cell_y: Field[[dims.EdgeDim, dims.E2CDim], float64],
    dual_normal_cell_x: Field[[dims.EdgeDim, dims.E2CDim], float64],
    dual_normal_cell_y: Field[[dims.EdgeDim, dims.E2CDim], float64],
    edge_center_lat: Field[[dims.EdgeDim], float64],
    edge_center_lon: Field[[dims.EdgeDim], float64],
    primal_normal_x: Field[[dims.EdgeDim], float64],
    primal_normal_y: Field[[dims.EdgeDim], float64],
    global_root: int32,
    global_level: int32,
    lowest_layer_thickness: float64,
    model_top_height: float64,
    stretch_factor: float64,
):
    logger.info(f"Using Device = {device}")

    global_grid_params = GlobalGridParams(root=global_root, level=global_level)

    # Edge geometry
    edge_params = geometry.EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=as_1D_sparse_field(primal_normal_vert_x, dims.ECVDim),
        primal_normal_vert_y=as_1D_sparse_field(primal_normal_vert_y, dims.ECVDim),
        dual_normal_vert_x=as_1D_sparse_field(dual_normal_vert_x, dims.ECVDim),
        dual_normal_vert_y=as_1D_sparse_field(dual_normal_vert_y, dims.ECVDim),
        primal_normal_cell_x=as_1D_sparse_field(primal_normal_cell_x, dims.ECDim),
        primal_normal_cell_y=as_1D_sparse_field(primal_normal_cell_y, dims.ECDim),
        dual_normal_cell_x=as_1D_sparse_field(dual_normal_cell_x, dims.ECDim),
        dual_normal_cell_y=as_1D_sparse_field(dual_normal_cell_y, dims.ECDim),
        edge_areas=edge_areas,
        f_e=f_e,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
    )

    # Cell geometry
    cell_params = geometry.CellParams.from_global_num_cells(
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
        num_levels=common.GLOBAL_STATE["icon_grid"].num_levels,
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
        zd_intcoef=zd_intcoef,
        # todo: for icon integration? flatten_first_two_dims(dims.CECDim, dims.KDim, field=zd_intcoef),
        zd_vertoffset=zd_vertoffset,
        # todo: for icon integration? flatten_first_two_dims(dims.CECDim, dims.KDim, field=zd_vertoffset),
        zd_diffcoef=zd_diffcoef,
    )

    # Interpolation state
    interpolation_state = DiffusionInterpolationState(
        e_bln_c_s=as_1D_sparse_field(e_bln_c_s, dims.CEDim),
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=as_1D_sparse_field(geofac_div, dims.CEDim),
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    # Initialize the diffusion granule
    common.GLOBAL_STATE["diffusion_granule"].init(
        grid=common.GLOBAL_STATE["icon_grid"],
        config=config,
        params=diffusion_params,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
    )


def diffusion_run(
    w: Field[[dims.CellDim, dims.KHalfDim], float64],
    vn: Field[[dims.EdgeDim, dims.KDim], float64],
    exner: Field[[dims.CellDim, dims.KDim], float64],
    theta_v: Field[[dims.CellDim, dims.KDim], float64],
    rho: Field[[dims.CellDim, dims.KDim], float64],
    hdef_ic: Field[[dims.CellDim, dims.KHalfDim], float64],
    div_ic: Field[[dims.CellDim, dims.KHalfDim], float64],
    dwdx: Field[[dims.CellDim, dims.KHalfDim], float64],
    dwdy: Field[[dims.CellDim, dims.KHalfDim], float64],
    dtime: float64,
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
        common.GLOBAL_STATE["diffusion_granule"].initial_run(
            diagnostic_state,
            prognostic_state,
            dtime,
        )
    else:
        common.GLOBAL_STATE["diffusion_granule"].run(
            prognostic_state=prognostic_state, diagnostic_state=diagnostic_state, dtime=dtime
        )
