# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# type: ignore

"""
Wrapper module for dycore granule.

Module contains a solve_nh_init and solve_nh_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""
import cProfile
import pstats

from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import float64, int32
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    PrepAdvection,
)
from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.dimension import (
    C2E2CODim,
    C2EDim,
    CEDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    ECDim,
    ECVDim,
    EdgeDim,
    KDim,
    KHalfDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.geometry import CellParams, EdgeParams
from icon4py.model.common.grid.icon import GlobalGridParams
from icon4py.model.common.grid.vertical import VerticalGrid, VerticalGridConfig
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import (
    as_1D_sparse_field,
    flatten_first_two_dims,
    zero_field,
)

from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.wrappers import common
from icon4pytools.py2fgen.wrappers.wrapper_dimension import (
    CellIndexDim,
    EdgeIndexDim,
    VertexIndexDim,
)


logger = setup_logger(__name__)

dycore_wrapper_state = {"granule": SolveNonhydro(), "profiler": cProfile.Profile()}


def profile_enable():
    dycore_wrapper_state["profiler"].enable()


def profile_disable():
    dycore_wrapper_state["profiler"].disable()
    stats = pstats.Stats(dycore_wrapper_state["profiler"])
    stats.dump_stats(f"{__name__}.profile")


def solve_nh_init(
    vct_a: Field[[KHalfDim], float64],
    vct_b: Field[[KHalfDim], float64],
    cell_areas: Field[[CellDim], float64],
    primal_normal_cell_x: Field[[EdgeDim, E2CDim], float64],
    primal_normal_cell_y: Field[[EdgeDim, E2CDim], float64],
    dual_normal_cell_x: Field[[EdgeDim, E2CDim], float64],
    dual_normal_cell_y: Field[[EdgeDim, E2CDim], float64],
    edge_areas: Field[[EdgeDim], float64],
    tangent_orientation: Field[[EdgeDim], float64],
    inverse_primal_edge_lengths: Field[[EdgeDim], float64],
    inverse_dual_edge_lengths: Field[[EdgeDim], float64],
    inverse_vertex_vertex_lengths: Field[[EdgeDim], float64],
    primal_normal_vert_x: Field[[EdgeDim, E2C2VDim], float64],
    primal_normal_vert_y: Field[[EdgeDim, E2C2VDim], float64],
    dual_normal_vert_x: Field[[EdgeDim, E2C2VDim], float64],
    dual_normal_vert_y: Field[[EdgeDim, E2C2VDim], float64],
    f_e: Field[[EdgeDim], float64],
    c_lin_e: Field[[EdgeDim, E2CDim], float64],
    c_intp: Field[[VertexDim, V2CDim], float64],
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float64],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float64],
    geofac_rot: Field[[VertexDim, V2EDim], float64],
    pos_on_tplane_e_1: Field[[EdgeDim, E2CDim], float64],
    pos_on_tplane_e_2: Field[[EdgeDim, E2CDim], float64],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float64],
    e_bln_c_s: Field[[CellDim, C2EDim], float64],
    rbf_coeff_1: Field[[VertexDim, V2EDim], float64],
    rbf_coeff_2: Field[[VertexDim, V2EDim], float64],
    geofac_div: Field[[CellDim, C2EDim], float64],
    geofac_n2s: Field[[CellDim, C2E2CODim], float64],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float64],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float64],
    nudgecoeff_e: Field[[EdgeDim], float64],
    bdy_halo_c: Field[[CellDim], bool],
    mask_prog_halo_c: Field[[CellDim], bool],
    rayleigh_w: Field[[KHalfDim], float64],
    exner_exfac: Field[[CellDim, KDim], float64],
    exner_ref_mc: Field[[CellDim, KDim], float64],
    wgtfac_c: Field[[CellDim, KHalfDim], float64],
    wgtfacq_c: Field[[CellDim, KDim], float64],
    inv_ddqz_z_full: Field[[CellDim, KDim], float64],
    rho_ref_mc: Field[[CellDim, KDim], float64],
    theta_ref_mc: Field[[CellDim, KDim], float64],
    vwind_expl_wgt: Field[[CellDim], float64],
    d_exner_dz_ref_ic: Field[[CellDim, KHalfDim], float64],
    ddqz_z_half: Field[[CellDim, KHalfDim], float64],
    theta_ref_ic: Field[[CellDim, KHalfDim], float64],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float64],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float64],
    rho_ref_me: Field[[EdgeDim, KDim], float64],
    theta_ref_me: Field[[EdgeDim, KDim], float64],
    ddxn_z_full: Field[[EdgeDim, KDim], float64],
    zdiff_gradp: Field[[EdgeDim, E2CDim, KDim], float64],
    vertoffset_gradp: Field[[EdgeDim, E2CDim, KDim], int32],
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float64],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float64],
    ddxt_z_full: Field[[EdgeDim, KDim], float64],
    wgtfac_e: Field[[EdgeDim, KHalfDim], float64],
    wgtfacq_e: Field[[EdgeDim, KDim], float64],
    vwind_impl_wgt: Field[[CellDim], float64],
    hmask_dd3d: Field[[EdgeDim], float64],
    scalfac_dd3d: Field[[KDim], float64],
    coeff1_dwdz: Field[[CellDim, KDim], float64],
    coeff2_dwdz: Field[[CellDim, KDim], float64],
    coeff_gradekin: Field[[EdgeDim, E2CDim], float64],
    c_owner_mask: Field[[CellDim], bool],
    cell_center_lat: Field[[CellDim], float64],
    cell_center_lon: Field[[CellDim], float64],
    edge_center_lat: Field[[EdgeDim], float64],
    edge_center_lon: Field[[EdgeDim], float64],
    primal_normal_x: Field[[EdgeDim], float64],
    primal_normal_y: Field[[EdgeDim], float64],
    rayleigh_damping_height: float64,
    itime_scheme: int32,
    iadv_rhotheta: int32,
    igradp_method: int32,
    ndyn_substeps: float64,
    rayleigh_type: int32,
    rayleigh_coeff: float64,
    divdamp_order: int32,
    is_iau_active: bool,
    iau_wgt_dyn: float64,
    divdamp_type: int32,
    divdamp_trans_start: float64,
    divdamp_trans_end: float64,
    l_vert_nested: bool,
    rhotheta_offctr: float64,
    veladv_offctr: float64,
    max_nudging_coeff: float64,
    divdamp_fac: float64,
    divdamp_fac2: float64,
    divdamp_fac3: float64,
    divdamp_fac4: float64,
    divdamp_z: float64,
    divdamp_z2: float64,
    divdamp_z3: float64,
    divdamp_z4: float64,
    lowest_layer_thickness: float64,
    model_top_height: float64,
    stretch_factor: float64,
    nflat_gradp: int32,
    num_levels: int32,
):
    config = NonHydrostaticConfig(
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        ndyn_substeps_var=ndyn_substeps,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        divdamp_order=divdamp_order,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
        divdamp_type=divdamp_type,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        l_vert_nested=l_vert_nested,
        rhotheta_offctr=rhotheta_offctr,
        veladv_offctr=veladv_offctr,
        max_nudging_coeff=max_nudging_coeff,
        divdamp_fac=divdamp_fac,
        divdamp_fac2=divdamp_fac2,
        divdamp_fac3=divdamp_fac3,
        divdamp_fac4=divdamp_fac4,
        divdamp_z=divdamp_z,
        divdamp_z2=divdamp_z2,
        divdamp_z3=divdamp_z3,
        divdamp_z4=divdamp_z4,
    )
    nonhydro_params = NonHydrostaticParams(config)

    # edge geometry
    edge_geometry = EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inverse_dual_edge_lengths,
        inverse_vertex_vertex_lengths=inverse_vertex_vertex_lengths,
        primal_normal_vert_x=as_1D_sparse_field(primal_normal_vert_x, ECVDim),
        primal_normal_vert_y=as_1D_sparse_field(primal_normal_vert_y, ECVDim),
        dual_normal_vert_x=as_1D_sparse_field(dual_normal_vert_x, ECVDim),
        dual_normal_vert_y=as_1D_sparse_field(dual_normal_vert_y, ECVDim),
        primal_normal_cell_x=as_1D_sparse_field(primal_normal_cell_x, ECDim),
        primal_normal_cell_y=as_1D_sparse_field(primal_normal_cell_y, ECDim),
        dual_normal_cell_x=as_1D_sparse_field(dual_normal_cell_x, ECDim),
        dual_normal_cell_y=as_1D_sparse_field(dual_normal_cell_y, ECDim),
        edge_areas=edge_areas,
        f_e=f_e,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
    )

    # datatest config CellParams
    cell_geometry = CellParams.from_global_num_cells(
        cell_center_lat=cell_center_lat,
        cell_center_lon=cell_center_lon,
        area=cell_areas,
        global_num_cells=dycore_wrapper_state["grid"].global_properties.num_cells,
        length_rescale_factor=1.0,
    )

    interpolation_state = InterpolationState(
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=as_1D_sparse_field(pos_on_tplane_e_1[:, 0:2], ECDim),
        pos_on_tplane_e_2=as_1D_sparse_field(pos_on_tplane_e_2[:, 0:2], ECDim),
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        e_bln_c_s=as_1D_sparse_field(e_bln_c_s, CEDim),
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=as_1D_sparse_field(geofac_div, CEDim),
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    metric_state_nonhydro = MetricStateNonHydro(
        bdy_halo_c=bdy_halo_c,
        mask_prog_halo_c=mask_prog_halo_c,
        rayleigh_w=rayleigh_w,
        exner_exfac=exner_exfac,
        exner_ref_mc=exner_ref_mc,
        wgtfac_c=wgtfac_c,
        wgtfacq_c=wgtfacq_c,
        inv_ddqz_z_full=inv_ddqz_z_full,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        vwind_expl_wgt=vwind_expl_wgt,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        theta_ref_ic=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        ddxn_z_full=ddxn_z_full,
        zdiff_gradp=flatten_first_two_dims(ECDim, KDim, field=zdiff_gradp),
        vertoffset_gradp=flatten_first_two_dims(ECDim, KDim, field=vertoffset_gradp),
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        vwind_impl_wgt=vwind_impl_wgt,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=as_1D_sparse_field(coeff_gradekin, ECDim),
    )

    # datatest config
    vertical_config = VerticalGridConfig(
        num_levels=num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=rayleigh_damping_height,
    )

    # datatest config, vertical parameters
    vertical_params = VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
        _min_index_flat_horizontal_grad_pressure=nflat_gradp,
    )

    dycore_wrapper_state["granule"].init(
        grid=dycore_wrapper_state["grid"],
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=c_owner_mask,
    )


def solve_nh_run(
    rho_now: Field[[CellDim, KDim], float64],
    rho_new: Field[[CellDim, KDim], float64],
    exner_now: Field[[CellDim, KDim], float64],
    exner_new: Field[[CellDim, KDim], float64],
    w_now: Field[[CellDim, KHalfDim], float64],
    w_new: Field[[CellDim, KHalfDim], float64],
    theta_v_now: Field[[CellDim, KDim], float64],
    theta_v_new: Field[[CellDim, KDim], float64],
    vn_now: Field[[EdgeDim, KDim], float64],
    vn_new: Field[[EdgeDim, KDim], float64],
    w_concorr_c: Field[[CellDim, KHalfDim], float64],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float64],
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], float64],
    ddt_w_adv_ntl1: Field[[CellDim, KHalfDim], float64],
    ddt_w_adv_ntl2: Field[[CellDim, KHalfDim], float64],
    theta_v_ic: Field[[CellDim, KHalfDim], float64],
    rho_ic: Field[[CellDim, KHalfDim], float64],
    exner_pr: Field[[CellDim, KDim], float64],
    exner_dyn_incr: Field[[CellDim, KDim], float64],
    ddt_exner_phy: Field[[CellDim, KDim], float64],
    grf_tend_rho: Field[[CellDim, KDim], float64],
    grf_tend_thv: Field[[CellDim, KDim], float64],
    grf_tend_w: Field[[CellDim, KHalfDim], float64],
    mass_fl_e: Field[[EdgeDim, KDim], float64],
    ddt_vn_phy: Field[[EdgeDim, KDim], float64],
    grf_tend_vn: Field[[EdgeDim, KDim], float64],
    vn_ie: Field[[EdgeDim, KHalfDim], float64],
    vt: Field[[EdgeDim, KDim], float64],
    mass_flx_me: Field[[EdgeDim, KDim], float64],
    mass_flx_ic: Field[[CellDim, KHalfDim], float64],
    vn_traj: Field[[EdgeDim, KDim], float64],
    dtime: float64,
    lprep_adv: bool,
    clean_mflx: bool,
    recompute: bool,
    linit: bool,
    divdamp_fac_o2: float64,
    ndyn_substeps: float64,
    idyn_timestep: int32,
    nnew: int32,
    nnow: int32,
):
    logger.info(f"Using Device = {settings.device}")

    prep_adv = PrepAdvection(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
        vol_flx_ic=zero_field(dycore_wrapper_state["grid"], CellDim, KDim, dtype=float),
    )

    diagnostic_state_nh = DiagnosticStateNonHydro(
        theta_v_ic=theta_v_ic,
        exner_pr=exner_pr,
        rho_ic=rho_ic,
        ddt_exner_phy=ddt_exner_phy,
        grf_tend_rho=grf_tend_rho,
        grf_tend_thv=grf_tend_thv,
        grf_tend_w=grf_tend_w,
        mass_fl_e=mass_fl_e,
        ddt_vn_phy=ddt_vn_phy,
        grf_tend_vn=grf_tend_vn,
        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
        ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
        ddt_w_adv_ntl1=ddt_w_adv_ntl1,
        ddt_w_adv_ntl2=ddt_w_adv_ntl2,
        vt=vt,
        vn_ie=vn_ie,
        w_concorr_c=w_concorr_c,
        rho_incr=None,  # sp.rho_incr,
        vn_incr=None,  # sp.vn_incr,
        exner_incr=None,  # sp.exner_incr,
        exner_dyn_incr=exner_dyn_incr,
    )

    prognostic_state_nnow = PrognosticState(
        w=w_now,
        vn=vn_now,
        theta_v=theta_v_now,
        rho=rho_now,
        exner=exner_now,
    )
    prognostic_state_nnew = PrognosticState(
        w=w_new,
        vn=vn_new,
        theta_v=theta_v_new,
        rho=rho_new,
        exner=exner_new,
    )
    prognostic_state_ls = [prognostic_state_nnow, prognostic_state_nnew]

    # adjust for Fortran indexes
    nnow = nnow - 1
    nnew = nnew - 1
    idyn_timestep = idyn_timestep - 1

    dycore_wrapper_state["granule"].time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_ls=prognostic_state_ls,
        prep_adv=prep_adv,
        divdamp_fac_o2=divdamp_fac_o2,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
        at_first_substep=idyn_timestep == 0,
        at_last_substep=idyn_timestep == (ndyn_substeps - 1),
    )


def grid_init(
    cell_starts: Field[[CellIndexDim], int32],
    cell_ends: Field[[CellIndexDim], int32],
    vertex_starts: Field[[VertexIndexDim], int32],
    vertex_ends: Field[[VertexIndexDim], int32],
    edge_starts: Field[[EdgeIndexDim], int32],
    edge_ends: Field[[EdgeIndexDim], int32],
    c2e: Field[[dims.CellDim, dims.C2EDim], int32],
    e2c: Field[[dims.EdgeDim, dims.E2CDim], int32],
    c2e2c: Field[[dims.CellDim, dims.C2E2CDim], int32],
    e2c2e: Field[[dims.EdgeDim, dims.E2C2EDim], int32],
    e2v: Field[[dims.EdgeDim, dims.E2VDim], int32],
    v2e: Field[[dims.VertexDim, dims.V2EDim], int32],
    v2c: Field[[dims.VertexDim, dims.V2CDim], int32],
    e2c2v: Field[[dims.EdgeDim, dims.E2C2VDim], int32],
    c2v: Field[[dims.CellDim, dims.C2VDim], int32],
    global_root: int32,
    global_level: int32,
    num_vertices: int32,
    num_cells: int32,
    num_edges: int32,
    vertical_size: int32,
    limited_area: bool,
):
    # todo: write this logic into template.py
    if isinstance(limited_area, int):
        limited_area = bool(limited_area)

    global_grid_params = GlobalGridParams(level=global_level, root=global_root)

    dycore_wrapper_state["grid"] = common.construct_icon_grid(
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
        grid_id="icon_grid",
        global_grid_params=global_grid_params,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=True if settings.device == "GPU" else False,
    )
