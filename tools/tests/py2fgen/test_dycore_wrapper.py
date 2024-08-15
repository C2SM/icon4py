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

import logging

from gt4py.next import as_field
from icon4py.model.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils.grid_utils import MCH_CH_R04B09_LEVELS

from icon4pytools.py2fgen.wrappers.dycore import solve_nh_init, solve_nh_run


logging.basicConfig(level=logging.INFO)


def test_solve_nh_wrapper():
    # Grid parameters
    num_cells = 20896
    num_edges = 31558
    num_verts = 10663
    num_levels = MCH_CH_R04B09_LEVELS
    num_c2ec2o = 4
    num_v2e = 6
    num_c2e = 3
    num_e2c2v = 4
    num_e2c = 2
    num_e2c2eo = 3  # todo: check
    mean_cell_area = 24907282236.708576

    # Other configuration parameters
    dtime = 10.0
    rayleigh_damping_height = 12500.0
    flat_height = 16000.0
    nflat_gradp = 59
    ndyn_substeps = 2.0
    jstep = 0

    # Nonhydrostatic configuration
    itime_scheme = 4  # itime scheme can only be 4
    iadv_rhotheta = 2
    igradp_method = 3
    rayleigh_type = 1
    rayleigh_coeff = 0.1
    divdamp_order = 24  # divdamp order can only be 24
    is_iau_active = False  # todo: ...
    iau_wgt_dyn = 0.5
    divdamp_fac_o2 = 0.5
    divdamp_type = 1
    divdamp_trans_start = 1000.0
    divdamp_trans_end = 2000.0
    l_vert_nested = False  # vertical nesting support is not implemented
    rhotheta_offctr = 1.0
    veladv_offctr = 1.0
    max_nudging_coeff = 0.1
    divdamp_fac = 1.0
    divdamp_fac2 = 2.0
    divdamp_fac3 = 3.0
    divdamp_fac4 = 4.0
    divdamp_z = 1.0
    divdamp_z2 = 2.0
    divdamp_z3 = 3.0
    divdamp_z4 = 4.0
    htop_moist_proc = 1000.0
    limited_area = True
    lprep_adv = False
    clean_mflx = True
    recompute = False
    linit = False

    # Input data - numpy
    rng = xp.random.default_rng()

    # The vct_a array must be set to the same values as the ones in ICON.
    # It represents the reference heights of vertical levels in meters, and many key vertical indices are derived from it.
    # Accurate computation of bounds relies on using the same vct_a values as those in ICON.
    vct_a = xp.asarray(
        [
            23000.0,
            20267.579776144084,
            18808.316862872744,
            17645.20947843258,
            16649.573524156993,
            15767.598849006221,
            14970.17804229092,
            14239.283693028447,
            13562.75820630252,
            12931.905058984285,
            12340.22824884565,
            11782.711681133735,
            11255.378878851721,
            10755.009592797565,
            10278.949589989745,
            9824.978499468381,
            9391.215299185755,
            8976.0490382992,
            8578.086969013575,
            8196.11499008041,
            7829.066987285794,
            7476.0007272129105,
            7136.078660578203,
            6808.552460051288,
            6492.750437928688,
            6188.067212417723,
            5893.955149722682,
            5609.917223280037,
            5335.501014932097,
            5070.293644636082,
            4813.9174616536775,
            4566.0263653241045,
            4326.302650484456,
            4094.4542934937413,
            3870.212611174545,
            3653.3302379273473,
            3443.5793766239703,
            3240.750287267658,
            3044.649984289428,
            2855.101119099911,
            2671.9410294241347,
            2495.0209412555105,
            2324.2053131841913,
            2159.371316580089,
            2000.4084488403732,
            1847.2182808658658,
            1699.7143443769428,
            1557.8221699649202,
            1421.479493379662,
            1290.63665617212,
            1165.2572384824416,
            1045.3189781024735,
            930.8150535208842,
            821.7558437251436,
            718.1713313259359,
            620.1144009054101,
            527.6654250683475,
            440.93877255014786,
            360.09231087410603,
            285.34182080238656,
            216.98400030452174,
            155.43579225710877,
            101.30847966961008,
            55.56948426298202,
            20.00000000000001,
            0.0,
        ]
    )

    theta_ref_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    wgtfac_c = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    e_bln_c_s = rng.uniform(low=0, high=1, size=(num_cells, num_c2e))
    geofac_div = rng.uniform(low=0, high=1, size=(num_cells, num_c2e))
    geofac_grg_x = rng.uniform(low=0, high=1, size=(num_cells, num_c2ec2o))
    geofac_grg_y = rng.uniform(low=0, high=1, size=(num_cells, num_c2ec2o))
    geofac_n2s = rng.uniform(low=0, high=1, size=(num_cells, num_c2ec2o))
    nudgecoeff_e = xp.zeros((num_edges,))
    rbf_coeff_1 = rng.uniform(low=0, high=1, size=(num_verts, num_v2e))
    rbf_coeff_2 = rng.uniform(low=0, high=1, size=(num_verts, num_v2e))
    w_now = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    w_new = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    vn_now = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    vn_new = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    exner_now = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    exner_new = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    theta_v_now = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    theta_v_new = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    rho_now = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    rho_new = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    dual_normal_cell_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    dual_normal_cell_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    dual_normal_vert_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    dual_normal_vert_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    primal_normal_cell_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    primal_normal_cell_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    primal_normal_vert_x = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    primal_normal_vert_y = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    tangent_orientation = rng.uniform(low=0, high=1, size=(num_edges))
    inverse_primal_edge_lengths = rng.uniform(low=0, high=1, size=(num_edges))
    inv_dual_edge_length = rng.uniform(low=0, high=1, size=(num_edges))
    inv_vert_vert_length = rng.uniform(low=0, high=1, size=(num_edges))
    edge_areas = rng.uniform(low=0, high=1, size=(num_edges))
    f_e = rng.uniform(low=0, high=1, size=(num_edges))
    cell_areas = rng.uniform(low=0, high=1, size=(num_cells))
    c_lin_e = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    c_intp = rng.uniform(low=0, high=1, size=(num_verts, num_e2c))
    e_flx_avg = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2eo))
    geofac_grdiv = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2eo))
    geofac_rot = rng.uniform(low=0, high=1, size=(num_verts, num_v2e))
    pos_on_tplane_e_1 = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    pos_on_tplane_e_2 = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    rbf_vec_coeff_e = rng.uniform(low=0, high=1, size=(num_edges, num_e2c2v))
    rayleigh_w = rng.uniform(low=0, high=1, size=(num_levels,))
    exner_exfac = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    exner_ref_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    wgtfacq_c_dsl = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    inv_ddqz_z_full = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    vwind_expl_wgt = rng.uniform(low=0, high=1, size=(num_cells,))
    d_exner_dz_ref_ic = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    ddqz_z_half = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    theta_ref_ic = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    d2dexdz2_fac1_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    d2dexdz2_fac2_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    rho_ref_me = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    theta_ref_me = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    ddxn_z_full = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    zdiff_gradp = rng.uniform(low=0, high=1, size=(num_edges, num_e2c, num_levels))
    vertoffset_gradp = xp.round(
        rng.uniform(low=0, high=1, size=(num_edges, num_e2c, num_levels))
    ).astype(xp.int32)
    ipeidx_dsl = rng.uniform(low=0, high=1, size=(num_edges, num_levels)) < 0.5
    pg_exdist = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    ddqz_z_full_e = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    ddxt_z_full = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    wgtfac_e = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    wgtfacq_e = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    vwind_impl_wgt = rng.uniform(low=0, high=1, size=(num_cells,))
    hmask_dd3d = rng.uniform(low=0, high=1, size=(num_edges,))
    scalfac_dd3d = rng.uniform(low=0, high=1, size=(num_levels,))
    coeff1_dwdz = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    coeff2_dwdz = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    coeff_gradekin = rng.uniform(low=0, high=1, size=(num_edges, num_e2c))
    rho_ref_mc = rng.uniform(low=0, high=1, size=(num_cells, num_levels))

    # Convert numpy arrays to gt4py fields
    vct_a = as_field((KDim,), vct_a)
    theta_ref_mc = as_field((CellDim, KDim), theta_ref_mc)
    wgtfac_c = as_field((CellDim, KDim), wgtfac_c)
    e_bln_c_s = as_field((CellDim, C2EDim), e_bln_c_s)
    geofac_div = as_field((CellDim, C2EDim), geofac_div)
    geofac_grg_x = as_field((CellDim, C2E2CODim), geofac_grg_x)
    geofac_grg_y = as_field((CellDim, C2E2CODim), geofac_grg_y)
    geofac_n2s = as_field((CellDim, C2E2CODim), geofac_n2s)
    nudgecoeff_e = as_field((EdgeDim,), nudgecoeff_e)
    rbf_coeff_1 = as_field((VertexDim, V2EDim), rbf_coeff_1)
    rbf_coeff_2 = as_field((VertexDim, V2EDim), rbf_coeff_2)
    w_now = as_field((CellDim, KDim), w_now)
    w_new = as_field((CellDim, KDim), w_new)
    vn_now = as_field((EdgeDim, KDim), vn_now)
    vn_new = as_field((EdgeDim, KDim), vn_new)
    exner_now = as_field((CellDim, KDim), exner_now)
    exner_new = as_field((CellDim, KDim), exner_new)
    theta_v_now = as_field((CellDim, KDim), theta_v_now)
    theta_v_new = as_field((CellDim, KDim), theta_v_new)
    rho_now = as_field((CellDim, KDim), rho_now)
    rho_new = as_field((CellDim, KDim), rho_new)
    dual_normal_cell_x = as_field((EdgeDim, E2CDim), dual_normal_cell_x)
    dual_normal_cell_y = as_field((EdgeDim, E2CDim), dual_normal_cell_y)
    dual_normal_vert_x = as_field((EdgeDim, E2C2VDim), dual_normal_vert_x)
    dual_normal_vert_y = as_field((EdgeDim, E2C2VDim), dual_normal_vert_y)
    primal_normal_cell_x = as_field((EdgeDim, E2CDim), primal_normal_cell_x)
    primal_normal_cell_y = as_field((EdgeDim, E2CDim), primal_normal_cell_y)
    primal_normal_vert_x = as_field((EdgeDim, E2C2VDim), primal_normal_vert_x)
    primal_normal_vert_y = as_field((EdgeDim, E2C2VDim), primal_normal_vert_y)
    tangent_orientation = as_field((EdgeDim,), tangent_orientation)
    inverse_primal_edge_lengths = as_field((EdgeDim,), inverse_primal_edge_lengths)
    inv_dual_edge_length = as_field((EdgeDim,), inv_dual_edge_length)
    inv_vert_vert_length = as_field((EdgeDim,), inv_vert_vert_length)
    edge_areas = as_field((EdgeDim,), edge_areas)
    f_e = as_field((EdgeDim,), f_e)
    cell_areas = as_field((CellDim,), cell_areas)
    c_lin_e = as_field((EdgeDim, E2CDim), c_lin_e)
    c_intp = as_field((VertexDim, V2CDim), c_intp)
    e_flx_avg = as_field((EdgeDim, E2C2EODim), e_flx_avg)
    geofac_grdiv = as_field((EdgeDim, E2C2EODim), geofac_grdiv)
    geofac_rot = as_field((VertexDim, V2EDim), geofac_rot)
    pos_on_tplane_e_1 = as_field((EdgeDim, E2CDim), pos_on_tplane_e_1)
    pos_on_tplane_e_2 = as_field((EdgeDim, E2CDim), pos_on_tplane_e_2)
    rbf_vec_coeff_e = as_field((EdgeDim, E2C2EDim), rbf_vec_coeff_e)
    rayleigh_w = as_field((KDim,), rayleigh_w)
    exner_exfac = as_field((CellDim, KDim), exner_exfac)
    exner_ref_mc = as_field((CellDim, KDim), exner_ref_mc)
    wgtfacq_c_dsl = as_field((CellDim, KDim), wgtfacq_c_dsl)
    inv_ddqz_z_full = as_field((CellDim, KDim), inv_ddqz_z_full)
    vwind_expl_wgt = as_field((CellDim,), vwind_expl_wgt)
    d_exner_dz_ref_ic = as_field((CellDim, KDim), d_exner_dz_ref_ic)
    ddqz_z_half = as_field((CellDim, KDim), ddqz_z_half)
    theta_ref_ic = as_field((CellDim, KDim), theta_ref_ic)
    d2dexdz2_fac1_mc = as_field((CellDim, KDim), d2dexdz2_fac1_mc)
    d2dexdz2_fac2_mc = as_field((CellDim, KDim), d2dexdz2_fac2_mc)
    rho_ref_me = as_field((EdgeDim, KDim), rho_ref_me)
    theta_ref_me = as_field((EdgeDim, KDim), theta_ref_me)
    ddxn_z_full = as_field((EdgeDim, KDim), ddxn_z_full)
    zdiff_gradp = as_field((EdgeDim, E2CDim, KDim), zdiff_gradp)
    vertoffset_gradp = as_field((EdgeDim, E2CDim, KDim), vertoffset_gradp)
    ipeidx_dsl = as_field((EdgeDim, KDim), ipeidx_dsl)
    pg_exdist = as_field((EdgeDim, KDim), pg_exdist)
    ddqz_z_full_e = as_field((EdgeDim, KDim), ddqz_z_full_e)
    ddxt_z_full = as_field((EdgeDim, KDim), ddxt_z_full)
    wgtfac_e = as_field((EdgeDim, KDim), wgtfac_e)
    wgtfacq_e = as_field((EdgeDim, KDim), wgtfacq_e)
    vwind_impl_wgt = as_field((CellDim,), vwind_impl_wgt)
    hmask_dd3d = as_field((EdgeDim,), hmask_dd3d)
    scalfac_dd3d = as_field((KDim,), scalfac_dd3d)
    coeff1_dwdz = as_field((CellDim, KDim), coeff1_dwdz)
    coeff2_dwdz = as_field((CellDim, KDim), coeff2_dwdz)
    coeff_gradekin = as_field((EdgeDim, E2CDim), coeff_gradekin)
    rho_ref_mc = as_field((CellDim, KDim), rho_ref_mc)

    # Create boolean arrays
    bdy_halo_c = as_field((CellDim,), rng.uniform(low=0, high=1, size=(num_cells,)) < 0.5)
    mask_prog_halo_c = as_field((CellDim,), rng.uniform(low=0, high=1, size=(num_cells,)) < 0.5)
    c_owner_mask = as_field((CellDim,), rng.uniform(low=0, high=1, size=(num_cells,)) < 0.5)

    solve_nh_init(
        vct_a=vct_a,
        nflat_gradp=nflat_gradp,
        num_levels=num_levels,
        mean_cell_area=mean_cell_area,
        cell_areas=cell_areas,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        edge_areas=edge_areas,
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        f_e=f_e,
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        e_bln_c_s=e_bln_c_s,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=geofac_div,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
        bdy_halo_c=bdy_halo_c,
        mask_prog_halo_c=mask_prog_halo_c,
        rayleigh_w=rayleigh_w,
        exner_exfac=exner_exfac,
        exner_ref_mc=exner_ref_mc,
        wgtfac_c=wgtfac_c,
        wgtfacq_c_dsl=wgtfacq_c_dsl,
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
        wgtfacq_e=wgtfacq_e,  # todo: wgtfacq_e_dsl
        vwind_impl_wgt=vwind_impl_wgt,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=coeff_gradekin,
        c_owner_mask=c_owner_mask,
        rayleigh_damping_height=rayleigh_damping_height,
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        ndyn_substeps=ndyn_substeps,
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
        htop_moist_proc=htop_moist_proc,
        limited_area=limited_area,
        flat_height=flat_height,
    )

    w_concorr_c = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    theta_v_ic = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    rho_ic = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    exner_pr = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    exner_dyn_incr = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    ddt_exner_phy = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    grf_tend_rho = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    grf_tend_thv = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    grf_tend_w = rng.uniform(low=0, high=1, size=(num_cells, num_levels + 1))
    mass_fl_e = rng.uniform(low=0, high=1, size=(num_edges, num_levels + 1))
    ddt_vn_phy = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    grf_tend_vn = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    vn_ie = rng.uniform(low=0, high=1, size=(num_edges, num_levels + 1))
    vt = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    mass_flx_me = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    mass_flx_ic = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    vn_traj = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    ddt_vn_apc_ntl1 = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    ddt_vn_apc_ntl2 = rng.uniform(low=0, high=1, size=(num_edges, num_levels))
    ddt_w_adv_ntl1 = rng.uniform(low=0, high=1, size=(num_cells, num_levels))
    ddt_w_adv_ntl2 = rng.uniform(low=0, high=1, size=(num_cells, num_levels))

    # Convert numpy arrays to gt4py fields
    w_concorr_c = as_field((CellDim, KDim), w_concorr_c)
    ddt_vn_apc_ntl1 = as_field((EdgeDim, KDim), ddt_vn_apc_ntl1)
    ddt_vn_apc_ntl2 = as_field((EdgeDim, KDim), ddt_vn_apc_ntl2)
    ddt_w_adv_ntl1 = as_field((CellDim, KDim), ddt_w_adv_ntl1)
    ddt_w_adv_ntl2 = as_field((CellDim, KDim), ddt_w_adv_ntl2)
    theta_v_ic = as_field((CellDim, KDim), theta_v_ic)
    rho_ic = as_field((CellDim, KDim), rho_ic)
    exner_pr = as_field((CellDim, KDim), exner_pr)
    exner_dyn_incr = as_field((CellDim, KDim), exner_dyn_incr)
    ddt_exner_phy = as_field((CellDim, KDim), ddt_exner_phy)
    grf_tend_rho = as_field((CellDim, KDim), grf_tend_rho)
    grf_tend_thv = as_field((CellDim, KDim), grf_tend_thv)
    grf_tend_w = as_field((CellDim, KDim), grf_tend_w)
    mass_fl_e = as_field((EdgeDim, KDim), mass_fl_e)
    ddt_vn_phy = as_field((EdgeDim, KDim), ddt_vn_phy)
    grf_tend_vn = as_field((EdgeDim, KDim), grf_tend_vn)
    vn_ie = as_field((EdgeDim, KDim), vn_ie)
    vt = as_field((EdgeDim, KDim), vt)
    mass_flx_me = as_field((EdgeDim, KDim), mass_flx_me)
    mass_flx_ic = as_field((CellDim, KDim), mass_flx_ic)
    vn_traj = as_field((EdgeDim, KDim), vn_traj)

    solve_nh_run(
        rho_now=rho_now,
        rho_new=rho_new,
        exner_now=exner_now,
        exner_new=exner_new,
        w_now=w_now,
        w_new=w_new,
        theta_v_now=theta_v_now,
        theta_v_new=theta_v_new,
        vn_now=vn_now,
        vn_new=vn_new,
        w_concorr_c=w_concorr_c,
        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
        ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
        ddt_w_adv_ntl1=ddt_w_adv_ntl1,
        ddt_w_adv_ntl2=ddt_w_adv_ntl2,
        theta_v_ic=theta_v_ic,
        rho_ic=rho_ic,
        exner_pr=exner_pr,
        exner_dyn_incr=exner_dyn_incr,
        ddt_exner_phy=ddt_exner_phy,
        grf_tend_rho=grf_tend_rho,
        grf_tend_thv=grf_tend_thv,
        grf_tend_w=grf_tend_w,
        mass_fl_e=mass_fl_e,
        ddt_vn_phy=ddt_vn_phy,
        grf_tend_vn=grf_tend_vn,
        vn_ie=vn_ie,
        vt=vt,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
        vn_traj=vn_traj,
        dtime=dtime,
        lprep_adv=lprep_adv,
        clean_mflx=clean_mflx,
        recompute=recompute,
        linit=linit,
        divdamp_fac_o2=divdamp_fac_o2,
        ndyn_substeps=ndyn_substeps,
        jstep=jstep,
    )
