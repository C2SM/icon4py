# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest import mock

import cffi
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, model_options, utils as common_utils
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.grid.vertical import VerticalGridConfig
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    helpers,
)
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils
from icon4py.tools.py2fgen.wrappers import (
    common as wrapper_common,
    dycore_wrapper,
)

from . import utils
from .test_grid_init import grid_init


logging.basicConfig(level=logging.INFO)


@pytest.fixture
def solve_nh_init(
    grid_savepoint,
    interpolation_savepoint,
    metrics_savepoint,
):
    itime_scheme = dycore_states.TimeSteppingScheme.MOST_EFFICIENT
    iadv_rhotheta = dycore_states.RhoThetaAdvectionType.MIURA
    igradp_method = dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    rayleigh_type = model_options.RayleighType.KLEMP
    rayleigh_coeff = 0.05
    divdamp_order = dycore_states.DivergenceDampingOrder.COMBINED
    is_iau_active = False
    iau_wgt_dyn = 1.0
    divdamp_type = 3
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    l_vert_nested = False
    rhotheta_offctr = -0.1
    veladv_offctr = 0.25
    max_nudging_coefficient = 0.375
    divdamp_fac = 0.004
    divdamp_fac2 = 0.004
    divdamp_fac3 = 0.004
    divdamp_fac4 = 0.004
    divdamp_z = 32500.0
    divdamp_z2 = 40000.0
    divdamp_z3 = 60000.0
    divdamp_z4 = 80000.0

    # vertical grid params
    num_levels = 65
    lowest_layer_thickness = 20.0
    model_top_height = 23000.0
    stretch_factor = 0.65
    rayleigh_damping_height = 12500.0

    # vertical params
    vct_a = test_utils.array_to_array_info(grid_savepoint.vct_a().ndarray)
    vct_b = test_utils.array_to_array_info(grid_savepoint.vct_b().ndarray)
    nflat_gradp = gtx.int32(
        grid_savepoint.nflat_gradp() + 1
    )  # undo the -1 to go back to Fortran value

    # metric state parameters
    bdy_halo_c = test_utils.array_to_array_info(metrics_savepoint.bdy_halo_c().ndarray)
    mask_prog_halo_c = test_utils.array_to_array_info(metrics_savepoint.mask_prog_halo_c().ndarray)
    rayleigh_w = test_utils.array_to_array_info(metrics_savepoint.rayleigh_w().ndarray)
    exner_exfac = test_utils.array_to_array_info(metrics_savepoint.exner_exfac().ndarray)
    exner_ref_mc = test_utils.array_to_array_info(metrics_savepoint.exner_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)
    wgtfacq_c = test_utils.array_to_array_info(metrics_savepoint.wgtfacq_c_dsl().ndarray)
    inv_ddqz_z_full = test_utils.array_to_array_info(metrics_savepoint.inv_ddqz_z_full().ndarray)
    rho_ref_mc = test_utils.array_to_array_info(metrics_savepoint.rho_ref_mc().ndarray)
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    vwind_expl_wgt = test_utils.array_to_array_info(metrics_savepoint.vwind_expl_wgt().ndarray)
    d_exner_dz_ref_ic = test_utils.array_to_array_info(
        metrics_savepoint.d_exner_dz_ref_ic().ndarray
    )
    ddqz_z_half = test_utils.array_to_array_info(metrics_savepoint.ddqz_z_half().ndarray)
    theta_ref_ic = test_utils.array_to_array_info(metrics_savepoint.theta_ref_ic().ndarray)
    d2dexdz2_fac1_mc = test_utils.array_to_array_info(metrics_savepoint.d2dexdz2_fac1_mc().ndarray)
    d2dexdz2_fac2_mc = test_utils.array_to_array_info(metrics_savepoint.d2dexdz2_fac2_mc().ndarray)
    rho_ref_me = test_utils.array_to_array_info(metrics_savepoint.rho_ref_me().ndarray)
    theta_ref_me = test_utils.array_to_array_info(metrics_savepoint.theta_ref_me().ndarray)
    ddxn_z_full = test_utils.array_to_array_info(metrics_savepoint.ddxn_z_full().ndarray)

    zdiff_gradp_field = metrics_savepoint._get_field(
        "zdiff_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim
    )
    zdiff_gradp = test_utils.array_to_array_info(zdiff_gradp_field.ndarray)

    vertoffset_gradp_field = metrics_savepoint._get_field(
        "vertoffset_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
    )
    vertoffset_gradp = test_utils.array_to_array_info(vertoffset_gradp_field.ndarray)

    pg_edgeidx_dsl = test_utils.array_to_array_info(metrics_savepoint.pg_edgeidx_dsl().ndarray)
    pg_exdist = test_utils.array_to_array_info(metrics_savepoint.pg_exdist().ndarray)
    ddqz_z_full_e = test_utils.array_to_array_info(metrics_savepoint.ddqz_z_full_e().ndarray)
    ddxt_z_full = test_utils.array_to_array_info(metrics_savepoint.ddxt_z_full().ndarray)
    wgtfac_e = test_utils.array_to_array_info(metrics_savepoint.wgtfac_e().ndarray)
    wgtfacq_e = test_utils.array_to_array_info(metrics_savepoint.wgtfacq_e_dsl(num_levels).ndarray)
    vwind_impl_wgt = test_utils.array_to_array_info(metrics_savepoint.vwind_impl_wgt().ndarray)
    hmask_dd3d = test_utils.array_to_array_info(metrics_savepoint.hmask_dd3d().ndarray)
    scalfac_dd3d = test_utils.array_to_array_info(metrics_savepoint.scalfac_dd3d().ndarray)
    coeff1_dwdz = test_utils.array_to_array_info(metrics_savepoint.coeff1_dwdz().ndarray)
    coeff2_dwdz = test_utils.array_to_array_info(metrics_savepoint.coeff2_dwdz().ndarray)

    coeff_gradekin_field = metrics_savepoint._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)
    coeff_gradekin = test_utils.array_to_array_info(coeff_gradekin_field.ndarray)

    # interpolation state parameters
    c_lin_e = test_utils.array_to_array_info(interpolation_savepoint.c_lin_e().ndarray)
    c_intp = test_utils.array_to_array_info(interpolation_savepoint.c_intp().ndarray)
    e_flx_avg = test_utils.array_to_array_info(interpolation_savepoint.e_flx_avg().ndarray)
    geofac_grdiv = test_utils.array_to_array_info(interpolation_savepoint.geofac_grdiv().ndarray)
    geofac_rot = test_utils.array_to_array_info(interpolation_savepoint.geofac_rot().ndarray)

    pos_on_tplane_e_1_field = interpolation_savepoint._get_field(
        "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_1 = test_utils.array_to_array_info(pos_on_tplane_e_1_field.ndarray)

    pos_on_tplane_e_2_field = interpolation_savepoint._get_field(
        "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_2 = test_utils.array_to_array_info(pos_on_tplane_e_2_field.ndarray)

    rbf_vec_coeff_e = test_utils.array_to_array_info(
        interpolation_savepoint.rbf_vec_coeff_e().ndarray
    )
    e_bln_c_s = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)
    rbf_coeff_1 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v1().ndarray)
    rbf_coeff_2 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v2().ndarray)
    geofac_div = test_utils.array_to_array_info(interpolation_savepoint.geofac_div().ndarray)
    geofac_n2s = test_utils.array_to_array_info(interpolation_savepoint.geofac_n2s().ndarray)
    geofac_grg_x = test_utils.array_to_array_info(interpolation_savepoint.geofac_grg()[0].ndarray)
    geofac_grg_y = test_utils.array_to_array_info(interpolation_savepoint.geofac_grg()[1].ndarray)
    nudgecoeff_e = test_utils.array_to_array_info(interpolation_savepoint.nudgecoeff_e().ndarray)

    # other params
    c_owner_mask = test_utils.array_to_array_info(grid_savepoint.c_owner_mask().ndarray)

    ffi = cffi.FFI()
    dycore_wrapper.solve_nh_init(
        ffi=ffi,
        perf_counters=None,
        vct_a=vct_a,
        vct_b=vct_b,
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
        zdiff_gradp=zdiff_gradp,
        vertoffset_gradp=vertoffset_gradp,
        ipeidx_dsl=pg_edgeidx_dsl,
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
        coeff_gradekin=coeff_gradekin,
        c_owner_mask=c_owner_mask,
        rayleigh_damping_height=rayleigh_damping_height,
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
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
        nudge_max_coeff=max_nudging_coefficient,
        divdamp_fac=divdamp_fac,
        divdamp_fac2=divdamp_fac2,
        divdamp_fac3=divdamp_fac3,
        divdamp_fac4=divdamp_fac4,
        divdamp_z=divdamp_z,
        divdamp_z2=divdamp_z2,
        divdamp_z3=divdamp_z3,
        divdamp_z4=divdamp_z4,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        nflat_gradp=nflat_gradp,
        num_levels=num_levels,
        backend=wrapper_common.BackendIntEnum.DEFAULT,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_dycore_wrapper_granule_inputs(
    grid_init,  # initializes the grid as side-effect
    istep_init,
    istep_exit,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    caplog,
    icon_grid,
    at_initial_timestep,
    backend,
):
    caplog.set_level(logging.DEBUG)

    # savepoints
    sp = savepoint_nonhydro_init

    # --- Granule input parameters for dycore init

    # non hydrostatic config parameters
    itime_scheme = dycore_states.TimeSteppingScheme.MOST_EFFICIENT
    iadv_rhotheta = dycore_states.RhoThetaAdvectionType.MIURA
    igradp_method = dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    ndyn_substeps = ndyn_substeps
    rayleigh_type = model_options.RayleighType.KLEMP
    rayleigh_coeff = 0.05
    divdamp_order = dycore_states.DivergenceDampingOrder.COMBINED
    is_iau_active = False
    iau_wgt_dyn = 1.0
    divdamp_type = 3
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    l_vert_nested = False
    rhotheta_offctr = -0.1
    veladv_offctr = 0.25
    max_nudging_coefficient = 0.375
    divdamp_fac = 0.004
    divdamp_fac2 = 0.004
    divdamp_fac3 = 0.004
    divdamp_fac4 = 0.004
    divdamp_z = 32500.0
    divdamp_z2 = 40000.0
    divdamp_z3 = 60000.0
    divdamp_z4 = 80000.0

    # vertical grid params
    num_levels = 65
    lowest_layer_thickness = 20.0
    model_top_height = 23000.0
    stretch_factor = 0.65
    rayleigh_damping_height = 12500.0

    # vertical params
    vct_a = test_utils.array_to_array_info(grid_savepoint.vct_a().ndarray)
    vct_b = test_utils.array_to_array_info(grid_savepoint.vct_b().ndarray)
    nflat_gradp = gtx.int32(
        grid_savepoint.nflat_gradp() + 1
    )  # undo the -1 to go back to Fortran value

    # other params
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    # metric state parameters
    bdy_halo_c = test_utils.array_to_array_info(metrics_savepoint.bdy_halo_c().ndarray)
    mask_prog_halo_c = test_utils.array_to_array_info(metrics_savepoint.mask_prog_halo_c().ndarray)
    rayleigh_w = test_utils.array_to_array_info(metrics_savepoint.rayleigh_w().ndarray)
    exner_exfac = test_utils.array_to_array_info(metrics_savepoint.exner_exfac().ndarray)
    exner_ref_mc = test_utils.array_to_array_info(metrics_savepoint.exner_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)
    wgtfacq_c = test_utils.array_to_array_info(metrics_savepoint.wgtfacq_c_dsl().ndarray)
    inv_ddqz_z_full = test_utils.array_to_array_info(metrics_savepoint.inv_ddqz_z_full().ndarray)
    rho_ref_mc = test_utils.array_to_array_info(metrics_savepoint.rho_ref_mc().ndarray)
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    vwind_expl_wgt = test_utils.array_to_array_info(metrics_savepoint.vwind_expl_wgt().ndarray)
    d_exner_dz_ref_ic = test_utils.array_to_array_info(
        metrics_savepoint.d_exner_dz_ref_ic().ndarray
    )
    ddqz_z_half = test_utils.array_to_array_info(metrics_savepoint.ddqz_z_half().ndarray)
    theta_ref_ic = test_utils.array_to_array_info(metrics_savepoint.theta_ref_ic().ndarray)
    d2dexdz2_fac1_mc = test_utils.array_to_array_info(metrics_savepoint.d2dexdz2_fac1_mc().ndarray)
    d2dexdz2_fac2_mc = test_utils.array_to_array_info(metrics_savepoint.d2dexdz2_fac2_mc().ndarray)
    rho_ref_me = test_utils.array_to_array_info(metrics_savepoint.rho_ref_me().ndarray)
    theta_ref_me = test_utils.array_to_array_info(metrics_savepoint.theta_ref_me().ndarray)
    ddxn_z_full = test_utils.array_to_array_info(metrics_savepoint.ddxn_z_full().ndarray)

    zdiff_gradp_field = metrics_savepoint._get_field(
        "zdiff_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim
    )
    zdiff_gradp = test_utils.array_to_array_info(zdiff_gradp_field.ndarray)

    vertoffset_gradp_field = metrics_savepoint._get_field(
        "vertoffset_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
    )
    vertoffset_gradp = test_utils.array_to_array_info(vertoffset_gradp_field.ndarray)

    pg_edgeidx_dsl = test_utils.array_to_array_info(metrics_savepoint.pg_edgeidx_dsl().ndarray)
    pg_exdist = test_utils.array_to_array_info(metrics_savepoint.pg_exdist().ndarray)
    ddqz_z_full_e = test_utils.array_to_array_info(metrics_savepoint.ddqz_z_full_e().ndarray)
    ddxt_z_full = test_utils.array_to_array_info(metrics_savepoint.ddxt_z_full().ndarray)
    wgtfac_e = test_utils.array_to_array_info(metrics_savepoint.wgtfac_e().ndarray)
    wgtfacq_e = test_utils.array_to_array_info(metrics_savepoint.wgtfacq_e_dsl(num_levels).ndarray)
    vwind_impl_wgt = test_utils.array_to_array_info(metrics_savepoint.vwind_impl_wgt().ndarray)
    hmask_dd3d = test_utils.array_to_array_info(metrics_savepoint.hmask_dd3d().ndarray)
    scalfac_dd3d = test_utils.array_to_array_info(metrics_savepoint.scalfac_dd3d().ndarray)
    coeff1_dwdz = test_utils.array_to_array_info(metrics_savepoint.coeff1_dwdz().ndarray)
    coeff2_dwdz = test_utils.array_to_array_info(metrics_savepoint.coeff2_dwdz().ndarray)

    coeff_gradekin_field = metrics_savepoint._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)
    coeff_gradekin = test_utils.array_to_array_info(coeff_gradekin_field.ndarray)

    # interpolation state parameters
    c_lin_e = test_utils.array_to_array_info(interpolation_savepoint.c_lin_e().ndarray)
    c_intp = test_utils.array_to_array_info(interpolation_savepoint.c_intp().ndarray)
    e_flx_avg = test_utils.array_to_array_info(interpolation_savepoint.e_flx_avg().ndarray)
    geofac_grdiv = test_utils.array_to_array_info(interpolation_savepoint.geofac_grdiv().ndarray)
    geofac_rot = test_utils.array_to_array_info(interpolation_savepoint.geofac_rot().ndarray)

    pos_on_tplane_e_1_field = interpolation_savepoint._get_field(
        "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_1 = test_utils.array_to_array_info(pos_on_tplane_e_1_field.ndarray)

    pos_on_tplane_e_2_field = interpolation_savepoint._get_field(
        "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_2 = test_utils.array_to_array_info(pos_on_tplane_e_2_field.ndarray)

    rbf_vec_coeff_e = test_utils.array_to_array_info(
        interpolation_savepoint.rbf_vec_coeff_e().ndarray
    )
    e_bln_c_s = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)
    rbf_coeff_1 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v1().ndarray)
    rbf_coeff_2 = test_utils.array_to_array_info(interpolation_savepoint.rbf_vec_coeff_v2().ndarray)
    geofac_div = test_utils.array_to_array_info(interpolation_savepoint.geofac_div().ndarray)
    geofac_n2s = test_utils.array_to_array_info(interpolation_savepoint.geofac_n2s().ndarray)
    geofac_grg_x = test_utils.array_to_array_info(interpolation_savepoint.geofac_grg()[0].ndarray)
    geofac_grg_y = test_utils.array_to_array_info(interpolation_savepoint.geofac_grg()[1].ndarray)
    nudgecoeff_e = test_utils.array_to_array_info(interpolation_savepoint.nudgecoeff_e().ndarray)

    # other params
    c_owner_mask = test_utils.array_to_array_info(grid_savepoint.c_owner_mask().ndarray)

    # --- Granule input parameters for dycore run
    second_order_divdamp_factor = sp.divdamp_fac_o2()

    # PrepAdvection
    vn_traj = test_utils.array_to_array_info(sp.vn_traj().ndarray)
    vol_flx_ic = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )  # TODO sp.vol_flx_ic()
    mass_flx_me = test_utils.array_to_array_info(sp.mass_flx_me().ndarray)
    mass_flx_ic = test_utils.array_to_array_info(sp.mass_flx_ic().ndarray)

    # Diagnostic state parameters
    max_vertical_cfl = 0.0
    theta_v_ic = test_utils.array_to_array_info(sp.theta_v_ic().ndarray)
    exner_pr = test_utils.array_to_array_info(sp.exner_pr().ndarray)
    rho_ic = test_utils.array_to_array_info(sp.rho_ic().ndarray)
    ddt_exner_phy = test_utils.array_to_array_info(sp.ddt_exner_phy().ndarray)
    grf_tend_rho = test_utils.array_to_array_info(sp.grf_tend_rho().ndarray)
    grf_tend_thv = test_utils.array_to_array_info(sp.grf_tend_thv().ndarray)
    grf_tend_w = test_utils.array_to_array_info(sp.grf_tend_w().ndarray)
    mass_fl_e = test_utils.array_to_array_info(sp.mass_fl_e().ndarray)
    ddt_vn_phy = test_utils.array_to_array_info(sp.ddt_vn_phy().ndarray)
    grf_tend_vn = test_utils.array_to_array_info(sp.grf_tend_vn().ndarray)
    ddt_vn_apc_ntl1 = test_utils.array_to_array_info(sp.ddt_vn_apc_pc(0).ndarray)
    ddt_vn_apc_ntl2 = test_utils.array_to_array_info(sp.ddt_vn_apc_pc(1).ndarray)
    ddt_w_adv_ntl1 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(0).ndarray)
    ddt_w_adv_ntl2 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(1).ndarray)
    vt = test_utils.array_to_array_info(sp.vt().ndarray)
    vn_ie = test_utils.array_to_array_info(sp.vn_ie().ndarray)
    vn_incr_field = data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim)
    vn_incr = test_utils.array_to_array_info(vn_incr_field.ndarray)
    rho_incr_field = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim)
    rho_incr = test_utils.array_to_array_info(rho_incr_field.ndarray)
    exner_incr_field = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim)
    exner_incr = test_utils.array_to_array_info(exner_incr_field.ndarray)
    w_concorr_c = test_utils.array_to_array_info(sp.w_concorr_c().ndarray)
    exner_dyn_incr = test_utils.array_to_array_info(sp.exner_dyn_incr().ndarray)

    # Prognostic state parameters
    w_now = test_utils.array_to_array_info(sp.w_now().ndarray)
    vn_now = test_utils.array_to_array_info(sp.vn_now().ndarray)
    theta_v_now = test_utils.array_to_array_info(sp.theta_v_now().ndarray)
    rho_now = test_utils.array_to_array_info(sp.rho_now().ndarray)
    exner_now = test_utils.array_to_array_info(sp.exner_now().ndarray)

    w_new = test_utils.array_to_array_info(sp.w_new().ndarray)
    vn_new = test_utils.array_to_array_info(sp.vn_new().ndarray)
    theta_v_new = test_utils.array_to_array_info(sp.theta_v_new().ndarray)
    rho_new = test_utils.array_to_array_info(sp.rho_new().ndarray)
    exner_new = test_utils.array_to_array_info(sp.exner_new().ndarray)

    # using fortran indices
    substep = substep_init

    # --- Expected objects that form inputs into init function ---
    expected_icon_grid = icon_grid
    expected_edge_geometry = grid_savepoint.construct_edge_geometry()
    expected_cell_geometry = grid_savepoint.construct_cell_geometry()
    # TODO fixture
    expected_interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        c_intp=interpolation_savepoint.c_intp(),
        e_flx_avg=interpolation_savepoint.e_flx_avg(),
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        geofac_rot=interpolation_savepoint.geofac_rot(),
        pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
        rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    expected_metric_state = dycore_states.MetricStateNonHydro(
        bdy_halo_c=metrics_savepoint.bdy_halo_c(),
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        time_extrapolation_parameter_for_exner=metrics_savepoint.exner_exfac(),
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        reference_rho_at_cells_on_model_levels=metrics_savepoint.rho_ref_mc(),
        reference_theta_at_cells_on_model_levels=metrics_savepoint.theta_ref_mc(),
        exner_w_explicit_weight_parameter=metrics_savepoint.vwind_expl_wgt(),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_savepoint.d_exner_dz_ref_ic(),
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        reference_theta_at_cells_on_half_levels=metrics_savepoint.theta_ref_ic(),
        d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
        reference_rho_at_edges_on_model_levels=metrics_savepoint.rho_ref_me(),
        reference_theta_at_edges_on_model_levels=metrics_savepoint.theta_ref_me(),
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
        pg_edgeidx_dsl=metrics_savepoint.pg_edgeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(num_levels),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )
    expected_vertical_config = VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    expected_vertical_params = v_grid.VerticalGrid(
        config=expected_vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    expected_config = utils.construct_solve_nh_config(experiment)
    expected_additional_parameters = solve_nh.NonHydrostaticParams(expected_config)

    # --- Expected objects that form inputs into run function ---
    expected_diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        # TODO (Chia Rui): read from serialized data
        max_vertical_cfl=0.0,
        tangential_wind=sp.vt(),
        vn_on_half_levels=sp.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=sp.w_concorr_c(),
        theta_v_at_cells_on_half_levels=sp.theta_v_ic(),
        perturbed_exner_at_cells_on_model_levels=sp.exner_pr(),
        rho_at_cells_on_half_levels=sp.rho_ic(),
        exner_tendency_due_to_slow_physics=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_flux_at_edges_on_model_levels=sp.mass_fl_e(),
        normal_wind_tendency_due_to_slow_physics_process=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            sp.ddt_vn_apc_pc(0), sp.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            sp.ddt_w_adv_pc(0), sp.ddt_w_adv_pc(1)
        ),
        rho_iau_increment=rho_incr_field,
        normal_wind_iau_increment=vn_incr_field,
        exner_iau_increment=exner_incr_field,
        exner_dynamical_increment=sp.exner_dyn_incr(),
    )
    prognostic_state_nnow = prognostics.PrognosticState(
        w=sp.w_now(),
        vn=sp.vn_now(),
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )
    prognostic_state_nnew = prognostics.PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )
    expected_prognostic_states = common_utils.TimeStepPair(
        prognostic_state_nnow, prognostic_state_nnew
    )

    expected_prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim
        ),  # TODO: sp.vol_flx_ic(),
    )
    expected_second_order_divdamp_factor = sp.divdamp_fac_o2()
    expected_dtime = sp.get_metadata("dtime").get("dtime")
    expected_lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    expected_at_first_substep = substep_init == 1
    expected_at_last_substep = substep_init == ndyn_substeps

    ffi = cffi.FFI()

    # --- Mock and Test SolveNonhydro.init ---
    with mock.patch(
        "icon4py.model.atmosphere.dycore.solve_nonhydro.SolveNonhydro.__init__",
        return_value=None,
    ) as mock_init:
        dycore_wrapper.solve_nh_init(
            ffi=ffi,
            perf_counters=None,
            vct_a=vct_a,
            vct_b=vct_b,
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
            zdiff_gradp=zdiff_gradp,
            vertoffset_gradp=vertoffset_gradp,
            ipeidx_dsl=pg_edgeidx_dsl,
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
            coeff_gradekin=coeff_gradekin,
            c_owner_mask=c_owner_mask,
            rayleigh_damping_height=rayleigh_damping_height,
            itime_scheme=itime_scheme,
            iadv_rhotheta=iadv_rhotheta,
            igradp_method=igradp_method,
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
            nudge_max_coeff=max_nudging_coefficient,
            divdamp_fac=divdamp_fac,
            divdamp_fac2=divdamp_fac2,
            divdamp_fac3=divdamp_fac3,
            divdamp_fac4=divdamp_fac4,
            divdamp_z=divdamp_z,
            divdamp_z2=divdamp_z2,
            divdamp_z3=divdamp_z3,
            divdamp_z4=divdamp_z4,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            nflat_gradp=nflat_gradp,
            num_levels=num_levels,
            backend=wrapper_common.BackendIntEnum.DEFAULT,
        )

        # Check input arguments to SolveNonhydro.init
        captured_args, captured_kwargs = mock_init.call_args

        # special case of grid._id as we do not use this arg in the wrapper as we cant pass strings from Fortran to the wrapper
        try:
            result, error_message = utils.compare_objects(
                captured_kwargs["grid"], expected_icon_grid
            )
            assert result, f"Grid comparison failed: {error_message}"
        except AssertionError as e:
            error_message = str(e)
            if "icon_grid != " not in error_message:
                raise
            else:
                pass

        result, error_message = utils.compare_objects(captured_kwargs["config"], expected_config)
        assert result, f"Config comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["params"], expected_additional_parameters
        )
        assert result, f"Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["metric_state_nonhydro"], expected_metric_state
        )
        assert result, f"Metric State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["interpolation_state"], expected_interpolation_state
        )
        assert result, f"Interpolation State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["vertical_params"], expected_vertical_params
        )
        assert result, f"Vertical Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["edge_geometry"], expected_edge_geometry
        )
        assert result, f"Edge Geometry comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["cell_geometry"], expected_cell_geometry
        )
        assert result, f"Cell Geometry comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["owner_mask"], grid_savepoint.c_owner_mask()
        )
        assert result, f"Owner Mask comparison failed: {error_message}"

    # --- Mock and Test SolveNonhydro.run ---
    with mock.patch(
        "icon4py.model.atmosphere.dycore.solve_nonhydro.SolveNonhydro.time_step"
    ) as mock_init:
        dycore_wrapper.solve_nh_run(
            ffi=ffi,
            perf_counters=None,
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
            vn_incr=vn_incr,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            mass_flx_me=mass_flx_me,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            vn_traj=vn_traj,
            dtime=dtime,
            max_vcfl=max_vertical_cfl,
            lprep_adv=lprep_adv,
            at_initial_timestep=at_initial_timestep,
            divdamp_fac_o2=second_order_divdamp_factor,
            ndyn_substeps_var=ndyn_substeps,
            idyn_timestep=substep,
        )

        # Check input arguments to SolveNonhydro.time_step
        captured_args, captured_kwargs = mock_init.call_args

        result, error_message = utils.compare_objects(
            captured_kwargs["diagnostic_state_nh"], expected_diagnostic_state_nh
        )
        assert result, f"Diagnostic State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["prognostic_states"], expected_prognostic_states
        )
        assert result, f"Prognostic State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["prep_adv"], expected_prep_adv
        )
        assert result, f"Prep Advection comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["second_order_divdamp_factor"], expected_second_order_divdamp_factor
        )
        assert result, f"Divdamp Factor comparison failed: {error_message}"

        result, error_message = utils.compare_objects(captured_kwargs["dtime"], expected_dtime)
        assert result, f"dtime comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["lprep_adv"], expected_lprep_adv
        )
        assert result, f"Prep Advection flag comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["at_first_substep"], expected_at_first_substep
        )
        assert result, f"First Substep comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["at_last_substep"], expected_at_last_substep
        )
        assert result, f"Last Substep comparison failed: {error_message}"


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
@pytest.mark.parametrize(
    "experiment,step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
def test_granule_solve_nonhydro_single_step_regional(
    grid_init,  # initializes the grid as side-effect
    solve_nh_init,  # initializes solve_nh as side-effect
    istep_init,
    istep_exit,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    savepoint_nonhydro_init,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    caplog,
    icon_grid,
    at_initial_timestep,
    backend,
):
    caplog.set_level(logging.DEBUG)

    # savepoints
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final

    # other params
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    # solve nh run parameters
    second_order_divdamp_factor = sp.divdamp_fac_o2()  # This is a scalar, don't convert

    # PrepAdvection
    vn_traj = test_utils.array_to_array_info(sp.vn_traj().ndarray)
    vol_flx_ic = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )
    mass_flx_me = test_utils.array_to_array_info(sp.mass_flx_me().ndarray)
    mass_flx_ic = test_utils.array_to_array_info(sp.mass_flx_ic().ndarray)

    # Diagnostic state parameters
    max_vertical_cfl = 0.0
    theta_v_ic = test_utils.array_to_array_info(sp.theta_v_ic().ndarray)
    exner_pr = test_utils.array_to_array_info(sp.exner_pr().ndarray)
    rho_ic = test_utils.array_to_array_info(sp.rho_ic().ndarray)
    ddt_exner_phy = test_utils.array_to_array_info(sp.ddt_exner_phy().ndarray)
    grf_tend_rho = test_utils.array_to_array_info(sp.grf_tend_rho().ndarray)
    grf_tend_thv = test_utils.array_to_array_info(sp.grf_tend_thv().ndarray)
    grf_tend_w = test_utils.array_to_array_info(sp.grf_tend_w().ndarray)
    mass_fl_e = test_utils.array_to_array_info(sp.mass_fl_e().ndarray)
    ddt_vn_phy = test_utils.array_to_array_info(sp.ddt_vn_phy().ndarray)
    grf_tend_vn = test_utils.array_to_array_info(sp.grf_tend_vn().ndarray)
    ddt_vn_apc_ntl1 = test_utils.array_to_array_info(sp.ddt_vn_apc_pc(0).ndarray)
    ddt_vn_apc_ntl2 = test_utils.array_to_array_info(sp.ddt_vn_apc_pc(1).ndarray)
    ddt_w_adv_ntl1 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(0).ndarray)
    ddt_w_adv_ntl2 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(1).ndarray)
    vt = test_utils.array_to_array_info(sp.vt().ndarray)
    vn_ie = test_utils.array_to_array_info(sp.vn_ie().ndarray)
    vn_incr = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim).ndarray
    )
    rho_incr = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )
    exner_incr = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )
    w_concorr_c = test_utils.array_to_array_info(sp.w_concorr_c().ndarray)
    exner_dyn_incr = test_utils.array_to_array_info(sp.exner_dyn_incr().ndarray)

    # Prognostic state parameters
    w_now = test_utils.array_to_array_info(sp.w_now().ndarray)
    vn_now = test_utils.array_to_array_info(sp.vn_now().ndarray)
    theta_v_now = test_utils.array_to_array_info(sp.theta_v_now().ndarray)
    rho_now = test_utils.array_to_array_info(sp.rho_now().ndarray)
    exner_now = test_utils.array_to_array_info(sp.exner_now().ndarray)

    w_new = test_utils.array_to_array_info(sp.w_new().ndarray)
    vn_new = test_utils.array_to_array_info(sp.vn_new().ndarray)
    theta_v_new = test_utils.array_to_array_info(sp.theta_v_new().ndarray)
    rho_new = test_utils.array_to_array_info(sp.rho_new().ndarray)
    exner_new = test_utils.array_to_array_info(sp.exner_new().ndarray)

    # using fortran indices
    substep = substep_init

    ffi = cffi.FFI()
    dycore_wrapper.solve_nh_run(
        ffi=ffi,
        perf_counters=None,
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
        vn_incr=vn_incr,
        rho_incr=rho_incr,
        exner_incr=exner_incr,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
        vn_traj=vn_traj,
        vol_flx_ic=vol_flx_ic,
        dtime=dtime,
        max_vcfl=max_vertical_cfl,
        lprep_adv=lprep_adv,
        at_initial_timestep=at_initial_timestep,
        divdamp_fac_o2=second_order_divdamp_factor,  # This is a scalar
        ndyn_substeps_var=ndyn_substeps,
        idyn_timestep=substep,
    )

    # Comparison asserts should now use py2fgen.as_array
    assert helpers.dallclose(
        py2fgen.as_array(ffi, theta_v_new, py2fgen.FLOAT64),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, exner_new, py2fgen.FLOAT64), sp_step_exit.exner_new().asnumpy()
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, vn_new, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-12,
        atol=1e-13,
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, rho_new, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, w_new, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, exner_dyn_incr, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize(
    "istep_init, substep_init, step_date_init, istep_exit, substep_exit, step_date_exit, vn_only, at_initial_timestep",
    [
        (1, 1, "2021-06-20T12:00:10.000", 2, 2, "2021-06-20T12:00:10.000", False, True),
        (1, 1, "2021-06-20T12:00:20.000", 2, 2, "2021-06-20T12:00:20.000", True, False),
    ],
)
@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
def test_granule_solve_nonhydro_multi_step_regional(
    grid_init,  # initializes the grid as side-effect
    solve_nh_init,  # initializes solve_nh as side-effect
    step_date_init,
    step_date_exit,
    istep_exit,
    substep_init,
    substep_exit,
    icon_grid,
    savepoint_nonhydro_init,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    experiment,
    ndyn_substeps,
    vn_only,  # TODO we don't use that value?
    at_initial_timestep,
    backend,
):
    # savepoints
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final

    # other params
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    # solve nh run parameters
    linit = sp.get_metadata("linit").get("linit")
    second_order_divdamp_factor = sp.divdamp_fac_o2()

    # PrepAdvection
    vn_traj = test_utils.array_to_array_info(sp.vn_traj().ndarray)
    vol_flx_ic = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )
    mass_flx_me = test_utils.array_to_array_info(sp.mass_flx_me().ndarray)
    mass_flx_ic = test_utils.array_to_array_info(sp.mass_flx_ic().ndarray)

    # Diagnostic state parameters
    max_vertical_cfl = 0.0
    theta_v_ic = test_utils.array_to_array_info(sp.theta_v_ic().ndarray)
    exner_pr = test_utils.array_to_array_info(sp.exner_pr().ndarray)
    rho_ic = test_utils.array_to_array_info(sp.rho_ic().ndarray)
    ddt_exner_phy = test_utils.array_to_array_info(sp.ddt_exner_phy().ndarray)
    grf_tend_rho = test_utils.array_to_array_info(sp.grf_tend_rho().ndarray)
    grf_tend_thv = test_utils.array_to_array_info(sp.grf_tend_thv().ndarray)
    grf_tend_w = test_utils.array_to_array_info(sp.grf_tend_w().ndarray)
    mass_fl_e = test_utils.array_to_array_info(sp.mass_fl_e().ndarray)
    ddt_vn_phy = test_utils.array_to_array_info(sp.ddt_vn_phy().ndarray)
    grf_tend_vn = test_utils.array_to_array_info(sp.grf_tend_vn().ndarray)
    ddt_vn_apc_ntl1 = test_utils.array_to_array_info(sp.ddt_vn_apc_pc(0).ndarray)
    ddt_vn_apc_ntl2 = test_utils.array_to_array_info(sp.ddt_vn_apc_pc(1).ndarray)
    if linit:
        ddt_w_adv_ntl1 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(0).ndarray)
        ddt_w_adv_ntl2 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(1).ndarray)
    else:
        ddt_w_adv_ntl1 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(1).ndarray)
        ddt_w_adv_ntl2 = test_utils.array_to_array_info(sp.ddt_w_adv_pc(0).ndarray)
    vt = test_utils.array_to_array_info(sp.vt().ndarray)
    vn_ie = test_utils.array_to_array_info(sp.vn_ie().ndarray)
    vn_incr = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim).ndarray
    )
    rho_incr = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )
    exner_incr = test_utils.array_to_array_info(
        data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim).ndarray
    )
    w_concorr_c = test_utils.array_to_array_info(sp.w_concorr_c().ndarray)
    exner_dyn_incr = test_utils.array_to_array_info(sp.exner_dyn_incr().ndarray)

    # Prognostic state parameters
    w_now = test_utils.array_to_array_info(sp.w_now().ndarray)
    vn_now = test_utils.array_to_array_info(sp.vn_now().ndarray)
    theta_v_now = test_utils.array_to_array_info(sp.theta_v_now().ndarray)
    rho_now = test_utils.array_to_array_info(sp.rho_now().ndarray)
    exner_now = test_utils.array_to_array_info(sp.exner_now().ndarray)

    w_new = test_utils.array_to_array_info(sp.w_new().ndarray)
    vn_new = test_utils.array_to_array_info(sp.vn_new().ndarray)
    theta_v_new = test_utils.array_to_array_info(sp.theta_v_new().ndarray)
    rho_new = test_utils.array_to_array_info(sp.rho_new().ndarray)
    exner_new = test_utils.array_to_array_info(sp.exner_new().ndarray)

    ffi = cffi.FFI()
    # use fortran indices in the driving loop to compute i_substep
    for i_substep in range(1, ndyn_substeps + 1):
        if not (at_initial_timestep and i_substep == 1):
            ddt_w_adv_ntl1, ddt_w_adv_ntl2 = ddt_w_adv_ntl2, ddt_w_adv_ntl1
        if not i_substep == 1:
            ddt_vn_apc_ntl1, ddt_vn_apc_ntl2 = ddt_vn_apc_ntl2, ddt_vn_apc_ntl1

        dycore_wrapper.solve_nh_run(
            ffi=ffi,
            perf_counters=None,
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
            vn_incr=vn_incr,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            mass_flx_me=mass_flx_me,
            mass_flx_ic=mass_flx_ic,
            vn_traj=vn_traj,
            vol_flx_ic=vol_flx_ic,
            dtime=dtime,
            max_vcfl=max_vertical_cfl,
            lprep_adv=lprep_adv,
            at_initial_timestep=at_initial_timestep,
            divdamp_fac_o2=second_order_divdamp_factor,
            ndyn_substeps_var=ndyn_substeps,
            idyn_timestep=i_substep,
        )

        w_new, w_now = w_now, w_new
        vn_new, vn_now = vn_now, vn_new
        theta_v_new, theta_v_now = theta_v_now, theta_v_new
        rho_new, rho_now = rho_now, rho_new
        exner_new, exner_now = exner_now, exner_new

    cell_start_lb_plus2 = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    edge_start_lb_plus4 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, rho_ic, py2fgen.FLOAT64)[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.rho_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, theta_v_ic, py2fgen.FLOAT64)[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.theta_v_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, mass_fl_e, py2fgen.FLOAT64)[edge_start_lb_plus4:, :],
        savepoint_nonhydro_exit.mass_fl_e().asnumpy()[edge_start_lb_plus4:, :],
        atol=5e-7,
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, mass_flx_me, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        atol=5e-7,
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, vn_traj, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        atol=1e-12,
    )

    # we compare against _now fields as _new and _now are switched internally in the granule.
    assert helpers.dallclose(
        py2fgen.as_array(ffi, theta_v_now, py2fgen.FLOAT64),
        sp_step_exit.theta_v_new().asnumpy(),
        atol=5e-7,
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, rho_now, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, exner_now, py2fgen.FLOAT64),
        sp_step_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, w_now, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        py2fgen.as_array(ffi, vn_now, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        atol=5e-13,
    )
    assert helpers.dallclose(
        py2fgen.as_array(ffi, exner_dyn_incr, py2fgen.FLOAT64),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )
