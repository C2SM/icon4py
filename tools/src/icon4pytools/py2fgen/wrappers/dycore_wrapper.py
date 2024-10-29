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

import gt4py.next as gtx
from gt4py.next import common as gt4py_common
from icon4py.model.atmosphere.dycore.nh_solve import solve_nonhydro
from icon4py.model.atmosphere.dycore.state_utils import states as nh_states
from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.decomposition import definitions
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
from icon4py.model.common.grid import icon
from icon4py.model.common.grid.geometry import CellParams, EdgeParams
from icon4py.model.common.grid.icon import GlobalGridParams
from icon4py.model.common.grid.vertical import VerticalGrid, VerticalGridConfig
from icon4py.model.common.settings import parallel_run
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import (
    as_1D_sparse_field,
    flatten_first_two_dims,
    zero_field,
)

from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.wrappers import common
from icon4pytools.py2fgen.wrappers.debug_utils import print_grid_decomp_info
from icon4pytools.py2fgen.wrappers.wrapper_dimension import (
    CellGlobalIndexDim,
    CellIndexDim,
    EdgeGlobalIndexDim,
    EdgeIndexDim,
    VertexGlobalIndexDim,
    VertexIndexDim,
)


logger = setup_logger(__name__)

dycore_wrapper_state = {
    "profiler": cProfile.Profile(),
}


def profile_enable():
    dycore_wrapper_state["profiler"].enable()


def profile_disable():
    dycore_wrapper_state["profiler"].disable()
    stats = pstats.Stats(dycore_wrapper_state["profiler"])
    stats.dump_stats(f"{__name__}.profile")


def solve_nh_init(
    vct_a: gt4py_common.Field[[KHalfDim], gtx.float64],
    vct_b: gt4py_common.Field[[KHalfDim], gtx.float64],
    cell_areas: gt4py_common.Field[[CellDim], gtx.float64],
    primal_normal_cell_x: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    primal_normal_cell_y: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    dual_normal_cell_x: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    dual_normal_cell_y: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    edge_areas: gt4py_common.Field[[EdgeDim], gtx.float64],
    tangent_orientation: gt4py_common.Field[[EdgeDim], gtx.float64],
    inverse_primal_edge_lengths: gt4py_common.Field[[EdgeDim], gtx.float64],
    inverse_dual_edge_lengths: gt4py_common.Field[[EdgeDim], gtx.float64],
    inverse_vertex_vertex_lengths: gt4py_common.Field[[EdgeDim], gtx.float64],
    primal_normal_vert_x: gt4py_common.Field[[EdgeDim, E2C2VDim], gtx.float64],
    primal_normal_vert_y: gt4py_common.Field[[EdgeDim, E2C2VDim], gtx.float64],
    dual_normal_vert_x: gt4py_common.Field[[EdgeDim, E2C2VDim], gtx.float64],
    dual_normal_vert_y: gt4py_common.Field[[EdgeDim, E2C2VDim], gtx.float64],
    f_e: gt4py_common.Field[[EdgeDim], gtx.float64],
    c_lin_e: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    c_intp: gt4py_common.Field[[VertexDim, V2CDim], gtx.float64],
    e_flx_avg: gt4py_common.Field[[EdgeDim, E2C2EODim], gtx.float64],
    geofac_grdiv: gt4py_common.Field[[EdgeDim, E2C2EODim], gtx.float64],
    geofac_rot: gt4py_common.Field[[VertexDim, V2EDim], gtx.float64],
    pos_on_tplane_e_1: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    pos_on_tplane_e_2: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    rbf_vec_coeff_e: gt4py_common.Field[[EdgeDim, E2C2EDim], gtx.float64],
    e_bln_c_s: gt4py_common.Field[[CellDim, C2EDim], gtx.float64],
    rbf_coeff_1: gt4py_common.Field[[VertexDim, V2EDim], gtx.float64],
    rbf_coeff_2: gt4py_common.Field[[VertexDim, V2EDim], gtx.float64],
    geofac_div: gt4py_common.Field[[CellDim, C2EDim], gtx.float64],
    geofac_n2s: gt4py_common.Field[[CellDim, C2E2CODim], gtx.float64],
    geofac_grg_x: gt4py_common.Field[[CellDim, C2E2CODim], gtx.float64],
    geofac_grg_y: gt4py_common.Field[[CellDim, C2E2CODim], gtx.float64],
    nudgecoeff_e: gt4py_common.Field[[EdgeDim], gtx.float64],
    bdy_halo_c: gt4py_common.Field[[CellDim], bool],
    mask_prog_halo_c: gt4py_common.Field[[CellDim], bool],
    rayleigh_w: gt4py_common.Field[[KHalfDim], gtx.float64],
    exner_exfac: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    exner_ref_mc: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    wgtfac_c: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    wgtfacq_c: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    inv_ddqz_z_full: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    rho_ref_mc: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    theta_ref_mc: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    vwind_expl_wgt: gt4py_common.Field[[CellDim], gtx.float64],
    d_exner_dz_ref_ic: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    ddqz_z_half: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    theta_ref_ic: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    d2dexdz2_fac1_mc: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    d2dexdz2_fac2_mc: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    rho_ref_me: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    theta_ref_me: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    ddxn_z_full: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    zdiff_gradp: gt4py_common.Field[[EdgeDim, E2CDim, KDim], gtx.float64],
    vertoffset_gradp: gt4py_common.Field[[EdgeDim, E2CDim, KDim], gtx.int32],
    ipeidx_dsl: gt4py_common.Field[[EdgeDim, KDim], bool],
    pg_exdist: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    ddqz_z_full_e: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    ddxt_z_full: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    wgtfac_e: gt4py_common.Field[[EdgeDim, KHalfDim], gtx.float64],
    wgtfacq_e: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    vwind_impl_wgt: gt4py_common.Field[[CellDim], gtx.float64],
    hmask_dd3d: gt4py_common.Field[[EdgeDim], gtx.float64],
    scalfac_dd3d: gt4py_common.Field[[KDim], gtx.float64],
    coeff1_dwdz: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    coeff2_dwdz: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    coeff_gradekin: gt4py_common.Field[[EdgeDim, E2CDim], gtx.float64],
    c_owner_mask: gt4py_common.Field[[CellDim], bool],
    cell_center_lat: gt4py_common.Field[[CellDim], gtx.float64],
    cell_center_lon: gt4py_common.Field[[CellDim], gtx.float64],
    edge_center_lat: gt4py_common.Field[[EdgeDim], gtx.float64],
    edge_center_lon: gt4py_common.Field[[EdgeDim], gtx.float64],
    primal_normal_x: gt4py_common.Field[[EdgeDim], gtx.float64],
    primal_normal_y: gt4py_common.Field[[EdgeDim], gtx.float64],
    rayleigh_damping_height: gtx.float64,
    itime_scheme: gtx.int32,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    ndyn_substeps: gtx.float64,
    rayleigh_type: gtx.int32,
    rayleigh_coeff: gtx.float64,
    divdamp_order: gtx.int32,
    is_iau_active: bool,
    iau_wgt_dyn: gtx.float64,
    divdamp_type: gtx.int32,
    divdamp_trans_start: gtx.float64,
    divdamp_trans_end: gtx.float64,
    l_vert_nested: bool,
    rhotheta_offctr: gtx.float64,
    veladv_offctr: gtx.float64,
    max_nudging_coeff: gtx.float64,
    divdamp_fac: gtx.float64,
    divdamp_fac2: gtx.float64,
    divdamp_fac3: gtx.float64,
    divdamp_fac4: gtx.float64,
    divdamp_z: gtx.float64,
    divdamp_z2: gtx.float64,
    divdamp_z3: gtx.float64,
    divdamp_z4: gtx.float64,
    lowest_layer_thickness: gtx.float64,
    model_top_height: gtx.float64,
    stretch_factor: gtx.float64,
    nflat_gradp: gtx.int32,
    num_levels: gtx.int32,
):
    if not isinstance(dycore_wrapper_state["grid"], icon.IconGrid):
        raise Exception("Need to initialise grid using grid_init_dycore before running solve_nh_init.")

    config = solve_nonhydro.NonHydrostaticConfig(
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
    nonhydro_params = solve_nonhydro.NonHydrostaticParams(config)

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

    interpolation_state = nh_states.InterpolationState(
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

    metric_state_nonhydro = nh_states.MetricStateNonHydro(
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
    rho_now: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    rho_new: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    exner_now: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    exner_new: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    w_now: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    w_new: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    theta_v_now: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    theta_v_new: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    vn_now: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    vn_new: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    w_concorr_c: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    ddt_vn_apc_ntl1: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    ddt_vn_apc_ntl2: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    ddt_w_adv_ntl1: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    ddt_w_adv_ntl2: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    theta_v_ic: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    rho_ic: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    exner_pr: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    exner_dyn_incr: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    ddt_exner_phy: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    grf_tend_rho: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    grf_tend_thv: gt4py_common.Field[[CellDim, KDim], gtx.float64],
    grf_tend_w: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    mass_fl_e: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    ddt_vn_phy: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    grf_tend_vn: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    vn_ie: gt4py_common.Field[[EdgeDim, KHalfDim], gtx.float64],
    vt: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    mass_flx_me: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    mass_flx_ic: gt4py_common.Field[[CellDim, KHalfDim], gtx.float64],
    vn_traj: gt4py_common.Field[[EdgeDim, KDim], gtx.float64],
    dtime: gtx.float64,
    lprep_adv: bool,
    clean_mflx: bool,
    recompute: bool,
    linit: bool,
    divdamp_fac_o2: gtx.float64,
    ndyn_substeps: gtx.float64,
    idyn_timestep: gtx.int32,
    nnew: gtx.int32,
    nnow: gtx.int32,
):
    logger.info(f"Using Device = {settings.device}")

    prep_adv = nh_states.PrepAdvection(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
        vol_flx_ic=zero_field(dycore_wrapper_state["grid"], CellDim, KDim, dtype=gtx.float64),
    )

    diagnostic_state_nh = nh_states.DiagnosticStateNonHydro(
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


def grid_init_dycore(
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

    if parallel_run:
        # Set MultiNodeExchange as exchange runtime
        processor_props, decomposition_info, exchange_runtime = common.construct_decomposition(
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
            dycore_wrapper_state["grid"],
            processor_props,
            decomposition_info,
            num_cells,
            num_edges,
            num_vertices,
        )
    else:
        exchange_runtime = definitions.SingleNodeExchange()

    # initialise the Diffusion granule
    dycore_wrapper_state["granule"] = solve_nonhydro.SolveNonhydro(
        backend=settings.backend, exchange=exchange_runtime
    )
