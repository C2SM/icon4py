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
import os
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
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalGridConfig, VerticalGridParams
from icon4py.model.common.settings import device
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.grid_utils import load_grid_from_file
from icon4py.model.common.test_utils.helpers import (
    as_1D_sparse_field,
    flatten_first_two_dims,
    zero_field,
)

from icon4pytools.common.logger import setup_logger
from icon4pytools.py2fgen.utils import get_grid_filename, get_icon_grid_loc


log = setup_logger(__name__)

# global diffusion object
solve_nonhydro = SolveNonhydro()

# global profiler object
profiler = cProfile.Profile()


def profile_enable():
    profiler.enable()


def profile_disable():
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(f"{__name__}.profile")


def solve_nh_init(
    vct_a: Field[[KDim], float64],
    nflat_gradp: int32,
    num_levels: int32,
    mean_cell_area: float64,
    cell_areas: Field[[CellDim], float64],
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
    c_owner_mask: Field[[CellDim], bool],
    rayleigh_damping_height: float64,
    itime_scheme: int32,
    iadv_rhotheta: int32,
    igradp_method: int32,
    ndyn_substeps: float64,
    rayleigh_type: int32,
    rayleigh_coeff: float64,
    divdamp_order: int32,  # the ICON default is 4,
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
    htop_moist_proc: float64,
    limited_area: bool,
    flat_height: float64,
):
    # ICON grid
    on_gpu = True if device.name == "GPU" else False

    grid_file_path = os.path.join(get_icon_grid_loc(), get_grid_filename())

    icon_grid = load_grid_from_file(
        grid_file=grid_file_path,
        num_levels=num_levels,
        on_gpu=on_gpu,
        limited_area=True if limited_area else False,
    )

    config = NonHydrostaticConfig(
        itime_scheme,
        iadv_rhotheta,
        igradp_method,
        ndyn_substeps,
        rayleigh_type,
        rayleigh_coeff,
        divdamp_order,
        is_iau_active,
        iau_wgt_dyn,
        divdamp_type,
        divdamp_trans_start,
        divdamp_trans_end,
        l_vert_nested,
        rhotheta_offctr,
        veladv_offctr,
        max_nudging_coeff,
        divdamp_fac,
        divdamp_fac2,
        divdamp_fac3,
        divdamp_fac4,
        divdamp_z,
        divdamp_z2,
        divdamp_z3,
        divdamp_z4,
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
        primal_normal_cell_x=as_1D_sparse_field(primal_normal_cell_x, ECDim),
        primal_normal_cell_y=as_1D_sparse_field(primal_normal_cell_y, ECDim),
        dual_normal_cell_x=as_1D_sparse_field(dual_normal_cell_x, ECDim),
        dual_normal_cell_y=as_1D_sparse_field(dual_normal_cell_y, ECDim),
        edge_areas=edge_areas,
        f_e=f_e,
    )

    # cell geometry
    cell_geometry = CellParams(
        area=cell_areas, mean_cell_area=mean_cell_area, length_rescale_factor=1.0
    )

    interpolation_state = InterpolationState(
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=as_1D_sparse_field(pos_on_tplane_e_1, ECDim),
        pos_on_tplane_e_2=as_1D_sparse_field(pos_on_tplane_e_2, ECDim),
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
        zdiff_gradp=flatten_first_two_dims(ECDim, KDim, field=zdiff_gradp),
        vertoffset_gradp=flatten_first_two_dims(ECDim, KDim, field=vertoffset_gradp),
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,  # todo: wgtfacq_e_dsl(num_k_lev),
        vwind_impl_wgt=vwind_impl_wgt,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=as_1D_sparse_field(coeff_gradekin, ECDim),
    )

    # vertical grid config
    vertical_config = VerticalGridConfig(
        num_levels=num_levels,
        rayleigh_damping_height=rayleigh_damping_height,
        htop_moist_proc=htop_moist_proc,
        flat_height=flat_height,
    )

    # vertical parameters
    vertical_params = VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=vct_a,
        vct_b=None,
        _min_index_flat_horizontal_grad_pressure=nflat_gradp,
    )

    solve_nonhydro.init(
        grid=icon_grid,
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
    ddt_w_adv_ntl1: Field[[CellDim, KDim], float64],
    ddt_w_adv_ntl2: Field[[CellDim, KDim], float64],
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
    mass_flx_me: Field[[EdgeDim, KDim], float64],
    mass_flx_ic: Field[[CellDim, KDim], float64],
    vn_traj: Field[[EdgeDim, KDim], float64],
    dtime: float64,
    lprep_adv: bool,
    clean_mflx: bool,
    recompute: bool,
    linit: bool,
    divdamp_fac_o2: float64,
    ndyn_substeps: float64,
    idyn_timestep: int32,
):
    log.info(f"Using Device = {device}")

    prep_adv = PrepAdvection(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
        vol_flx_ic=zero_field(solve_nonhydro.grid, CellDim, KDim, dtype=float),
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

    solve_nonhydro.time_step(
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
