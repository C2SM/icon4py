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
Wrapper module for diffusion granule.

Module contains a diffusion_init and diffusion_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""
import cProfile
import os
import pstats

from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import float64, int32

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    PrepAdvection, MetricStateNonHydro, InterpolationState,
)
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.grid.horizontal import (
    CellParams,
    EdgeParams,
    HorizontalMarkerIndex,
)
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    ECVDim,
    EdgeDim,
    KDim,
    KHalfDim,
    V2EDim,
    VertexDim, C2VDim, E2VDim, V2CDim, E2C2EODim,
)
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.settings import device, parallel_run
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.grid_utils import load_grid_from_file
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, flatten_first_two_dims
from icon4py.model.common.test_utils.parallel_helpers import check_comm_size

from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.utils import get_grid_filename, get_icon_grid_loc
from icon4pytools.py2fgen.wrapper_utils.debug_output import print_grid_decomp_info
from icon4pytools.py2fgen.wrapper_utils.dimension import (
    CellIndexDim,
    EdgeIndexDim,
    SingletonDim,
    SpecialADim,
    SpecialBDim,
    SpecialCDim,
    VertexIndexDim,
)
from icon4pytools.py2fgen.wrapper_utils.grid_utils import (
    construct_decomposition,
    construct_icon_grid,
)


log = setup_logger(__name__)

# global diffusion object
solve_nonhydro: Diffusion = None

# global profiler object
profiler = cProfile.Profile()


def profile_enable():
    profiler.enable()


def profile_disable():
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(f"{__name__}.profile")


def solve_nh_init(
    vct_a: Field[[KHalfDim], float64],
    vct_b: Field[[KHalfDim], float64],
    nrdmax: int32,
    nflat_gradp: int32,
    nflatlev: int32,
    num_cells: int32,
    num_edges: int32,
    num_verts: int32,
    num_levels: int32,
    nshift_total: int32,
    nshift: int32,
    mean_cell_area: float64,
    cell_areas: Field[[CellDim], float64],
    cell_center_lat: Field[[CellDim], float64] ,
    cell_center_lon: Field[[CellDim], float64],
    c2e: Field[[CellDim, SingletonDim, C2EDim], int32],
    c2e2c: Field[[CellDim, SingletonDim, C2E2CDim], int32],
    c2v: Field[[CellDim, SingletonDim, C2VDim], int32],
    cells_start_index: Field[[CellIndexDim], int32],
    cells_end_index: Field[[CellIndexDim], int32],
    refin_ctrl: Field[[CellDim], int32],
    c_owner_mask: Field[[CellDim], bool],
    c_glb_index: Field[[SpecialADim], int32],
    primal_normal_cell_x: Field[[EdgeDim, E2CDim], float64],
    primal_normal_cell_y: Field[[EdgeDim, E2CDim], float64],
    dual_normal_cell_x: Field[[EdgeDim, E2CDim], float64],
    dual_normal_cell_y: Field[[EdgeDim, E2CDim], float64],
    edge_areas: Field[[EdgeDim], float64],
    tangent_orientation: Field[[EdgeDim], float64],
    inverse_primal_edge_lengths: Field[[EdgeDim], float64],
    inv_dual_edge_length: Field[[EdgeDim], float64],
    inv_vert_vert_length: Field[[EdgeDim], float64],
    primal_normal_vert_x: Field[[EdgeDim, E2C2VDim], float64],
    primal_normal_vert_y: Field[[EdgeDim, E2C2VDim], float64],
    dual_normal_vert_x: Field[[EdgeDim, E2C2VDim], float64],
    dual_normal_vert_y: Field[[EdgeDim, E2C2VDim], float64],
    e2v: Field[[EdgeDim, SingletonDim, E2VDim], int32],
    e2c2v: Field[[EdgeDim, SingletonDim, E2C2VDim], int32],
    edges_center_lat: Field[[EdgeDim], float64] ,
    edges_center_lon: Field[[EdgeDim], float64] ,
    e2c: Field[[EdgeDim, SingletonDim, E2CDim], int32],
    e_owner_mask: Field[[EdgeDim], bool],
    e_glb_index: Field[[SpecialBDim], int32],
    edge_start_index: Field[[EdgeIndexDim], int32],
    edge_end_index: Field[[EdgeIndexDim], int32],
    e2c2e:,
    f_e: Field[[EdgeDim], float64],
    v2e: Field[[VertexDim, SingletonDim, V2EDim], int32],
    v2c: Field[[VertexDim, SingletonDim, V2CDim], int32],
    vert_start_index: Field[[VertexIndexDim], int32],
    vert_end_index: Field[[VertexIndexDim], int32],
    v_owner_mask: Field[[VertexDim], bool],
    v_glb_index: Field[[SpecialCDim], int32],
    c_lin_e: Field[[EdgeDim, E2CDim], float64],
    c_intp: Field[[VertexDim, V2CDim], float64],
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float64],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float64],
    geofac_rot: Field[[VertexDim, V2EDim], float64],
    pos_on_tplane_e_1: Field[[EdgeDim, E2CDim], float64],
    pos_on_tplane_e_2: Field[[EdgeDim, E2CDim], float64],
    rbf_vec_coeff_e: Field[[EdgeDim, RBFVecDim], float64],
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
    rayleigh_w: Field[[KDim], float64],
    exner_exfac: Field[[CellDim, KDim], float64],
    exner_ref_mc: Field[[CellDim, KDim], float64],
    wgtfac_c: Field[[CellDim, KDim], float64],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float64],
    inv_ddqz_z_full: Field[[CellDim, KDim], float64],
    rho_ref_mc: Field[[CellDim, KDim], float64],
    theta_ref_mc: Field[[CellDim, KDim], float64],
    vwind_expl_wgt: Field[[CellDim], float64],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float64],
    ddqz_z_half: Field[[CellDim, KDim], float64],
    theta_ref_ic: Field[[CellDim, KDim], float64],
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
    wgtfac_e: Field[[EdgeDim, KDim], float64],
    wgtfacq_e: Field[[EdgeDim, KDim], float64],
    vwind_impl_wgt: Field[[CellDim], float64],
    hmask_dd3d: Field[[EdgeDim], float64],
    scalfac_dd3d: Field[[KDim], float64],
    coeff1_dwdz: Field[[CellDim, KDim], float64],
    coeff2_dwdz: Field[[CellDim, KDim], float64],
    coeff_gradekin: Field[[EdgeDim, E2CDim], float64],
    comm_id: int32,
):
# ICON grid
    on_gpu = True if device.name == "GPU" else False

    if parallel_run:
        icon_grid = construct_icon_grid( #TODO add more for solve_nh
            cells_start_index,
            cells_end_index,
            vert_start_index,
            vert_end_index,
            edge_start_index,
            edge_end_index,
            num_cells,
            num_edges,
            num_verts,
            num_levels,
            c2e,
            c2e2c,
            v2e,
            e2c2v,
            e2c,
            True,
            on_gpu,
        )

        processor_props, decomposition_info, exchange = construct_decomposition(
            c_glb_index,
            e_glb_index,
            v_glb_index,
            c_owner_mask,
            e_owner_mask,
            v_owner_mask,
            num_cells,
            num_edges,
            num_verts,
            num_levels,
            comm_id,
        )

        check_comm_size(processor_props)

        print_grid_decomp_info(
            icon_grid, processor_props, decomposition_info, num_cells, num_edges, num_verts
        )
    else:
        grid_file_path = os.path.join(get_icon_grid_loc(), get_grid_filename())

        icon_grid = load_grid_from_file(
            grid_file=grid_file_path,
            num_levels=num_levels,
            on_gpu=on_gpu,
            limited_area=True if limited_area else False,
        )

    nonhydro_params = NonHydrostaticParams(config)

    # Edge geometry
    edge_geometry = EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=as_1D_sparse_field(primal_normal_vert_x, ECVDim),
        primal_normal_vert_y=as_1D_sparse_field(primal_normal_vert_y, ECVDim),
        dual_normal_vert_x=as_1D_sparse_field(dual_normal_vert_x, ECVDim),
        dual_normal_vert_y=as_1D_sparse_field(dual_normal_vert_y, ECVDim),
        primal_normal_cell_x=as_1D_sparse_field(primal_normal_cell_x, ECVDim),
        primal_normal_cell_y=as_1D_sparse_field(primal_normal_cell_y, ECVDim),
        dual_normal_cell_x=as_1D_sparse_field(dual_normal_cell_x, ECVDim),
        dual_normal_cell_y=as_1D_sparse_field(dual_normal_cell_y, ECVDim),
        edge_areas=edge_areas,
        f_e=f_e,
        edge_center_lat=edges_center_lat,
        edge_center_lon=edges_center_lon,
        primal_normal_x=,
        primal_normal_y=,
    )

    # cell geometry
    cell_geometry = CellParams(cell_center_lat=cell_center_lat,cell_center_lon=cell_center_lon,area=cell_areas, global_num_cells=global_num_cells,)

    interpolation_state = InterpolationState(
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
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
        wgtfacq_c=wgtfacq_c_dsl,
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
        zdiff_gradp=zdiff_gradp,
        vertoffset_gradp=vertoffset_gradp,
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e_dsl(num_k_lev),
        vwind_impl_wgt=vwind_impl_wgt,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=coeff_gradekin,
    )

    # vertical parameters
    vertical_params = VerticalModelParams(
        vct_a=vct_a,
        rayleigh_damping_height=rayleigh_damping_height,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
    )
    global solve_nonhydro
    if parallel_run:
        solve_nonhydro = SolveNonhydro(exchange=exchange)
    else:
        solve_nonhydro = SolveNonhydro()

    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_c_owner_mask,
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
    ddt_vn_apc_pc: Field[[CellDim, KDim,1_3Dim], float64],
    ddt_w_adv_pc: Field[[CellDim, KDim,1_3Dim], float64],
    theta_v_ic: Field[[CellDim, KHalfDim], float64],
    rho_ic: Field[[CellDim, KHalfDim], float64],
    exner_pr: Field[[CellDim, KDim], float64],
    exner_dyn_incr: Field[[CellDim, KDim], float64],
    ddt_exner_phy: Field[[CellDim, KDim], float64],
    grf_tend_rho: Field[[CellDim, KDim], float64],
    grf_tend_thv: Field[[CellDim, KDim], float64],
    grf_tend_w: Field[[CellDim, KHalfDim], float64],
    mass_fl_e: Field[[EdgeDim, KHalfDim], float64],
    ddt_vn_phy: Field[[EdgeDim, KDim], float64],
    grf_tend_vn: Field[[EdgeDim, KDim], float64],
    vn_ie: Field[[EdgeDim, KHalfDim], float64],
    vt: Field[[EdgeDim, KDim], float64],
    mass_flx_me: Field[[EdgeDim, KDim], float64], #change
    mass_flx_ic: Field[[EdgeDim, KDim], float64], #change
    vn_traj: Field[[EdgeDim, KDim], float64], #change
    vn_incr: Field[[EdgeDim, KDim], float64], #change
    rho_incr: Field[[EdgeDim, KDim], float64], #change
    exner_incr: Field[[EdgeDim, KDim], float64], #change
    dtime: float64,
    lprep_adv: bool,
    clean_mflx: bool,
    recompute: bool,
    linit: bool,
    divdamp_fac_o2: float64,
    limited_area: bool,
):
    log.info(f"Using Device = {device}")

    # ICON grid
    on_gpu = True if device.name == "GPU" else False

    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj,
        mass_flx_me=sp.mass_flx_me,
        mass_flx_ic=sp.mass_flx_ic,
        vol_flx_ic=zero_field(icon_grid, CellDim, KDim, dtype=float),
    )

    nnow = 0
    nnew = 1

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
        ddt_vn_apc_ntl1=ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=ddt_w_adv_pc(2),
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

    initial_divdamp_fac =


    global solve_nonhydro
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_ls=prognostic_state_ls,
        prep_adv=prep_adv,
        divdamp_fac_o2=initial_divdamp_fac,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
        at_first_substep=jstep_init == 0,
        at_last_substep=jstep_init == (ndyn_substeps - 1),
    )
    prognostic_state_nnew = prognostic_state_ls[1]
