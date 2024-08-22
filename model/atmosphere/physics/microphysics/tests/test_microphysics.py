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

import pytest

import numpy as np
from icon4py.model.common.dimension import CellDim, KDim
import gt4py.next as gtx
from icon4py.model.atmosphere.physics.microphysics import single_moment_six_class_gscp_graupel as graupel
from icon4py.model.atmosphere.physics.microphysics import single_moment_six_class_gscp_graupel_ref as graupel_ref
from icon4py.model.atmosphere.physics.microphysics import saturation_adjustment
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics, tracer_state as tracers
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.parametrize(
    "experiment, model_top_height,, damping_height, stretch_factor, date",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "48"),
        #(dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "52"),
        #(dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "56"),
    ],
)
def test_graupel(
    experiment,
    model_top_height,
    damping_height,
    stretch_factor,
    date,
    data_provider,
    grid_savepoint,
    metrics_savepoint,
    icon_grid,
    lowest_layer_thickness,
):
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
    )

    init_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_init()
    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_exit(date=date)

    dtime = entry_savepoint.dt_microphysics()

    assert vertical_params.kstart_moist == entry_savepoint.kstart_moist() - 1

    tracer_state = tracers.TracerState(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qr=entry_savepoint.qr(),
        qi=entry_savepoint.qi(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=entry_savepoint.rho(),
        vn=None,
        w=None,
        exner=None,
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=entry_savepoint.temperature(),
        pressure=entry_savepoint.pres(),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    saturation_adjustment_config = saturation_adjustment.SaturationAdjustmentConfig()

    print(init_savepoint.iautocon())
    print(init_savepoint.ceff_min())
    print(init_savepoint.vz0i())
    print(init_savepoint.icesedi_exp())
    print(init_savepoint.v0snow())
    print(init_savepoint.mu_rain())
    print(init_savepoint.rain_n0_factor())
    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        do_saturation_adjustment=False,
        liquid_autoconversion_option=init_savepoint.iautocon(),
        ice_stickeff_min=init_savepoint.ceff_min(),
        ice_v0=init_savepoint.vz0i(),
        ice_sedi_density_factor_exp=init_savepoint.icesedi_exp(),
        snow_v0=init_savepoint.v0snow(),
        rain_mu=init_savepoint.mu_rain(),
        rain_n0=init_savepoint.rain_n0_factor(),
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        saturation_adjust_config=saturation_adjustment_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
    )

    assert dallclose(graupel.icon_graupel_params.qmin, init_savepoint.qmin())
    assert dallclose(graupel.icon_graupel_params.eps, init_savepoint.eps())
    assert dallclose(graupel.icon_graupel_params.snow_m0, init_savepoint.ams())
    assert dallclose(graupel_microphysics.ccs[0], init_savepoint.ccsrim(), atol=1.e-8)
    assert dallclose(graupel_microphysics.ccs[1], init_savepoint.ccsagg(), atol=1.e-8)
    assert dallclose(graupel.icon_graupel_params.ccsaxp, init_savepoint.ccsaxp())
    assert dallclose(graupel.icon_graupel_params.ccsdep, init_savepoint.ccsdep(), atol=1.e-7)
    assert dallclose(graupel.icon_graupel_params.ccsdxp, init_savepoint.ccsdxp())
    assert dallclose(graupel_microphysics.ccs[2], init_savepoint.ccsvel(), atol=1.e-8)
    assert dallclose(graupel.icon_graupel_params.ccsvxp, init_savepoint.ccsvxp())
    assert dallclose(graupel.icon_graupel_params.ccslam, init_savepoint.ccslam(), atol=1.e-10)
    assert dallclose(graupel.icon_graupel_params.ccslxp, init_savepoint.ccslxp())
    assert dallclose(graupel.icon_graupel_params.ccshi1, init_savepoint.ccshi1())
    assert dallclose(graupel.icon_graupel_params.ccdvtp, init_savepoint.ccdvtp())
    assert dallclose(graupel.icon_graupel_params.ccidep, init_savepoint.ccidep())
    assert dallclose(graupel_microphysics.rain_vel_coef[2], init_savepoint.cevxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[3], init_savepoint.cev(), atol=1.e-10)
    assert dallclose(graupel_microphysics.rain_vel_coef[4], init_savepoint.bevxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[5], init_savepoint.bev())
    assert dallclose(graupel_microphysics.rain_vel_coef[0], init_savepoint.vzxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[1], init_savepoint.vz0r(), atol=1.e-10)

    start_cell_nudging = icon_grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.nudging(CellDim)
    )

    end_cell_local = icon_grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.local(CellDim))
    backend = run_gtfn_cached

    # TODO (Chia Rui): remove this slicing process when either the scan operator can be run on the gtfn backend or running on embedded backend is faster
    max_index = np.unravel_index(np.abs(entry_savepoint.qv().ndarray - exit_savepoint.qv().ndarray).argmax(), exit_savepoint.qv().ndarray.shape)
    print("max", max_index, vertical_params.kstart_moist)

    cell_lower_limit = max_index[0] - 3
    cell_upper_limit = max_index[0] + 4
    cell_size = cell_upper_limit - cell_lower_limit
    vertical_size = icon_grid.num_levels

    slice_qv = gtx.as_field((CellDim, KDim), entry_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_qc = gtx.as_field((CellDim, KDim), entry_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_qi = gtx.as_field((CellDim, KDim), entry_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_qr = gtx.as_field((CellDim, KDim), entry_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_qs = gtx.as_field((CellDim, KDim), entry_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_qg = gtx.as_field((CellDim, KDim), entry_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit, :])

    slice_t = gtx.as_field((CellDim, KDim), entry_savepoint.temperature().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_pres = gtx.as_field((CellDim, KDim), entry_savepoint.pres().ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_rho = gtx.as_field((CellDim, KDim), entry_savepoint.rho().ndarray[cell_lower_limit:cell_upper_limit, :])

    print(entry_savepoint.qnc().ndarray.shape)
    qnc_2d = entry_savepoint.qnc().ndarray
    qnc = gtx.as_field((CellDim, KDim), np.repeat(np.expand_dims(qnc_2d, axis=-1), icon_grid.num_levels, axis=1))
    slice_qnc = gtx.as_field((CellDim, KDim), qnc.ndarray[cell_lower_limit:cell_upper_limit, :])
    #slice_qnc = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    print(slice_qnc.ndarray.shape)
    for k in range(icon_grid.num_levels):
        print(slice_qnc.ndarray[:,k])
    slice_ddqz_z_full = gtx.as_field((CellDim, KDim), graupel_microphysics.metric_state.ddqz_z_full.ndarray[cell_lower_limit:cell_upper_limit, :])

    slice_qv_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qc_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qi_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qr_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qs_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qg_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))

    slice_t_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))

    slice_rhoqrv_old_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_rhoqsv_old_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_rhoqgv_old_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_rhoqiv_old_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_r = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))

    # for using the old version
    dist_cldtop = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    rho_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    crho1o2_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    crhofac_qi_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    snow_sed0_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    qvsw_kup = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    k_lev = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=gtx.int32))
    new_temperature_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    new_qv_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    new_qc_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    new_qr_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    new_qi_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    new_qs_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    new_qg_old_graupel = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_szdep_v2i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_szsub_v2i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_snucl_v2i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_scfrz_c2i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_simlt_i2c = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_sicri_i2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_sidep_v2i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_sdaut_i2s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_saggs_i2s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_saggg_i2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_siaut_i2s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_ssmlt_s2r = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_srims_c2s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_ssdep_v2s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_scosg_s2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_sgmlt_g2r = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_srcri_r2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_sgdep_v2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_srfrz_r2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    ser_graupel_srimg_c2g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))

    '''
    graupel_microphysics.metric_state.ddqz_z_full,
    diagnostic_state.temperature,
    diagnostic_state.pressure,
    prognostic_state.rho,
    tracer_state.qv,
    tracer_state.qc,
    tracer_state.qi,
    tracer_state.qr,
    tracer_state.qs,
    tracer_state.qg,
    graupel_microphysics.qnc,
    graupel_microphysics.temperature_tendency,
    graupel_microphysics.qv_tendency,
    graupel_microphysics.qc_tendency,
    graupel_microphysics.qi_tendency,
    graupel_microphysics.qr_tendency,
    graupel_microphysics.qs_tendency,
    graupel_microphysics.qg_tendency,
    graupel_microphysics.rhoqrv_old_kup,
    graupel_microphysics.rhoqsv_old_kup,
    graupel_microphysics.rhoqgv_old_kup,
    graupel_microphysics.rhoqiv_old_kup,
    graupel_microphysics.vnew_r,
    graupel_microphysics.vnew_s,
    graupel_microphysics.vnew_g,
    graupel_microphysics.vnew_i,


    # new graupel scan
    graupel._icon_graupel_scan(
        vertical_params.kstart_moist,
        gtx.int32(icon_grid.num_levels) - gtx.int32(1),
        graupel_microphysics.config.liquid_autoconversion_option,
        graupel_microphysics.config.snow_intercept_option,
        graupel_microphysics.config.is_isochoric,
        graupel_microphysics.config.use_constant_water_heat_capacity,
        graupel_microphysics.config.ice_stickeff_min,
        graupel_microphysics.config.ice_v0,
        graupel_microphysics.config.ice_sedi_density_factor_exp,
        graupel_microphysics.config.snow_v0,
        graupel_microphysics.ccs[0],
        graupel_microphysics.ccs[1],
        graupel_microphysics.ccs[2],
        graupel_microphysics.rain_vel_coef[0],
        graupel_microphysics.rain_vel_coef[1],
        graupel_microphysics.rain_vel_coef[2],
        graupel_microphysics.rain_vel_coef[3],
        graupel_microphysics.rain_vel_coef[4],
        graupel_microphysics.rain_vel_coef[5],
        graupel_microphysics.sed_dens_factor_coef[0],
        graupel_microphysics.sed_dens_factor_coef[1],
        graupel_microphysics.sed_dens_factor_coef[2],
        dtime,
        slice_ddqz_z_full,
        slice_t,
        slice_pres,
        slice_rho,
        slice_qv,
        slice_qc,
        slice_qi,
        slice_qr,
        slice_qs,
        slice_qg,
        slice_qnc,
        out=(
            slice_t_t,
            slice_qv_t,
            slice_qc_t,
            slice_qi_t,
            slice_qr_t,
            slice_qs_t,
            slice_qg_t,
            new_qv_old_graupel,
            new_qc_old_graupel,
            new_qi_old_graupel,
            new_qr_old_graupel,
            new_qs_old_graupel,
            new_qg_old_graupel,
            slice_rhoqrv_old_kup,
            slice_rhoqsv_old_kup,
            slice_rhoqgv_old_kup,
            slice_rhoqiv_old_kup,
            slice_vnew_r,
            slice_vnew_s,
            slice_vnew_g,
            slice_vnew_i,
            dist_cldtop,
            rho_kup,
            crho1o2_kup,
            crhofac_qi_kup,
            snow_sed0_kup,
            qvsw_kup,
            k_lev,
        ),
        offset_provider={},
    )
    '''
    use_new = True
    if use_new:
        # new graupel
        graupel.icon_graupel(
            vertical_params.kstart_moist,
            gtx.int32(icon_grid.num_levels) - gtx.int32(1),
            graupel_microphysics.config.liquid_autoconversion_option,
            graupel_microphysics.config.snow_intercept_option,
            graupel_microphysics.config.is_isochoric,
            graupel_microphysics.config.use_constant_water_heat_capacity,
            graupel_microphysics.config.ice_stickeff_min,
            graupel_microphysics.config.ice_v0,
            graupel_microphysics.config.ice_sedi_density_factor_exp,
            graupel_microphysics.config.snow_v0,
            *graupel_microphysics.ccs,
            *graupel_microphysics.rain_vel_coef,
            *graupel_microphysics.sed_dens_factor_coef,
            dtime,
            slice_ddqz_z_full,
            slice_t,
            slice_pres,
            slice_rho,
            slice_qv,
            slice_qc,
            slice_qi,
            slice_qr,
            slice_qs,
            slice_qg,
            slice_qnc,
            slice_t_t,
            slice_qv_t,
            slice_qc_t,
            slice_qi_t,
            slice_qr_t,
            slice_qs_t,
            slice_qg_t,
            slice_rhoqrv_old_kup,
            slice_rhoqsv_old_kup,
            slice_rhoqgv_old_kup,
            slice_rhoqiv_old_kup,
            slice_vnew_r,
            slice_vnew_s,
            slice_vnew_g,
            slice_vnew_i,
            # dist_cldtop,
            # rho_kup,
            # crho1o2_kup,
            # crhofac_qi_kup,
            # snow_sed0_kup,
            # qvsw_kup,
            # k_lev,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(cell_size),
            # vertical_start=vertical_params.kstart_moist,
            vertical_start=gtx.int32(0),
            vertical_end=vertical_size,
            offset_provider={},
        )
        new_temperature = entry_savepoint.temperature().ndarray[cell_lower_limit:cell_upper_limit,
                          0:icon_grid.num_levels] + slice_t_t.ndarray * dtime
        new_qv = entry_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit,
                 0:icon_grid.num_levels] + slice_qv_t.ndarray * dtime
        new_qc = entry_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit,
                 0:icon_grid.num_levels] + slice_qc_t.ndarray * dtime
        new_qr = entry_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit,
                 0:icon_grid.num_levels] + slice_qr_t.ndarray * dtime
        new_qi = entry_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit,
                 0:icon_grid.num_levels] + slice_qi_t.ndarray * dtime
        new_qs = entry_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit,
                 0:icon_grid.num_levels] + slice_qs_t.ndarray * dtime
        new_qg = entry_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit,
                 0:icon_grid.num_levels] + slice_qg_t.ndarray * dtime

    else:
        # old graupel
        graupel_ref._graupel_scan(
            vertical_params.kstart_moist,
            icon_grid.num_levels - 1,
            dtime,
            slice_ddqz_z_full,
            slice_t,
            slice_pres,
            slice_rho,
            slice_qv,
            slice_qc,
            slice_qi,
            slice_qr,
            slice_qs,
            slice_qg,
            slice_qnc,
            True,  # l_cv
            0,  # ithermo_water
            out=(
                new_temperature_old_graupel,
                new_qv_old_graupel,
                new_qc_old_graupel,
                new_qi_old_graupel,
                new_qr_old_graupel,
                new_qs_old_graupel,
                new_qg_old_graupel,
                # used in graupel scheme, do not output to outer world
                slice_rhoqrv_old_kup,
                slice_rhoqsv_old_kup,
                slice_rhoqgv_old_kup,
                slice_rhoqiv_old_kup,
                slice_vnew_r,
                slice_vnew_s,
                slice_vnew_g,
                slice_vnew_i,
                dist_cldtop,
                rho_kup,
                crho1o2_kup,
                crhofac_qi_kup,
                snow_sed0_kup,  # Cvz0s_kup
                qvsw_kup,
                k_lev,
                ser_graupel_szdep_v2i,
                ser_graupel_szsub_v2i,
                ser_graupel_snucl_v2i,
                ser_graupel_scfrz_c2i,
                ser_graupel_simlt_i2c,
                ser_graupel_sicri_i2g,
                ser_graupel_sidep_v2i,
                ser_graupel_sdaut_i2s,
                ser_graupel_saggs_i2s,
                ser_graupel_saggg_i2g,
                ser_graupel_siaut_i2s,
                ser_graupel_ssmlt_s2r,
                ser_graupel_srims_c2s,
                ser_graupel_ssdep_v2s,
                ser_graupel_scosg_s2g,
                ser_graupel_sgmlt_g2r,
                ser_graupel_srcri_r2g,
                ser_graupel_sgdep_v2g,
                ser_graupel_srfrz_r2g,
                ser_graupel_srimg_c2g
            ),
            offset_provider={}
        )
        new_temperature = new_temperature_old_graupel.ndarray
        new_qv = new_qv_old_graupel.ndarray
        new_qc = new_qc_old_graupel.ndarray
        new_qr = new_qr_old_graupel.ndarray
        new_qi = new_qi_old_graupel.ndarray
        new_qs = new_qs_old_graupel.ndarray
        new_qg = new_qg_old_graupel.ndarray


    '''
    graupel.icon_graupel_flux_above_ground(
        graupel_microphysics.config.do_latent_heat_nudging,
        prognostic_state.rho,
        tracer_state.qr,
        tracer_state.qs,
        tracer_state.qg,
        tracer_state.qi,
        graupel_microphysics.qr_tendency,
        graupel_microphysics.qs_tendency,
        graupel_microphysics.qg_tendency,
        graupel_microphysics.qi_tendency,
        graupel_microphysics.vnew_r,
        graupel_microphysics.vnew_s,
        graupel_microphysics.vnew_g,
        graupel_microphysics.vnew_i,
        graupel_microphysics.rain_precipitation_flux,
        graupel_microphysics.snow_precipitation_flux,
        graupel_microphysics.graupel_precipitation_flux,
        graupel_microphysics.ice_precipitation_flux,
        graupel_microphysics.total_precipitation_flux,
        start_cell_nudging,
        end_cell_local,
        gtx.int32(vertical_params.kstart_moist),
        gtx.int32(icon_grid.num_levels) - gtx.int32(1),
        offset_provider={},
    )

    graupel.icon_graupel_flux_ground(
        graupel_microphysics.config.do_latent_heat_nudging,
        prognostic_state.rho,
        tracer_state.qr,
        tracer_state.qs,
        tracer_state.qg,
        tracer_state.qi,
        graupel_microphysics.qr_tendency,
        graupel_microphysics.qs_tendency,
        graupel_microphysics.qg_tendency,
        graupel_microphysics.qi_tendency,
        graupel_microphysics.rhoqrv_old_kup,
        graupel_microphysics.rhoqsv_old_kup,
        graupel_microphysics.rhoqgv_old_kup,
        graupel_microphysics.rhoqiv_old_kup,
        graupel_microphysics.vnew_r,
        graupel_microphysics.vnew_s,
        graupel_microphysics.vnew_g,
        graupel_microphysics.vnew_i,
        graupel_microphysics.rain_precipitation_flux,
        graupel_microphysics.snow_precipitation_flux,
        graupel_microphysics.graupel_precipitation_flux,
        graupel_microphysics.ice_precipitation_flux,
        graupel_microphysics.total_precipitation_flux,
        start_cell_nudging,
        end_cell_local,
        gtx.int32(icon_grid.num_levels) - gtx.int32(1),
        gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    for k in range(icon_grid.num_levels):
        if np.any(np.where(graupel_microphysics.temperature_tendency.ndarray[:, k] > 0.0, True, False)):
            print(k, " is nonzero.", np.max(graupel_microphysics.temperature_tendency.ndarray[:, k]))
            for i in range(max_index[0]-10, max_index[0]+10):
                print("debug: ", i, graupel_microphysics.temperature_tendency.ndarray[i, k])
            break
    for i in range(cell_size):
        for k in range(icon_grid.num_levels):
            print(i,k,slice_t_t.ndarray[i, k])
    '''


    slice_result_t = exit_savepoint.temperature().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    slice_result_qv = exit_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    slice_result_qc = exit_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    slice_result_qi = exit_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    slice_result_qr = exit_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    slice_result_qs = exit_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    slice_result_qg = exit_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    relative_error_t = np.where(slice_result_t > 1.e-15, np.abs(new_temperature - slice_result_t)/slice_result_t, 0.0)
    relative_error_qv = np.where(np.abs(slice_result_qv) > 1.e-10, np.abs(
        (new_qv - slice_result_qv) / slice_result_qv), 0.0)
    relative_error_qc = np.where(slice_result_qc > 1.e-10, np.abs(
        (new_qc - slice_result_qc) / slice_result_qc), 0.0)
    relative_error_qi = np.where(slice_result_qi > 1.e-10, np.abs(
        (new_qi - slice_result_qi) / slice_result_qi), 0.0)
    relative_error_qr = np.where(slice_result_qr > 1.e-10, np.abs(
        (new_qr - slice_result_qr) / slice_result_qr), 0.0)
    relative_error_qs = np.where(slice_result_qs > 1.e-10, np.abs(
        (new_qs - slice_result_qs) / slice_result_qs), 0.0)
    relative_error_qg = np.where(slice_result_qg > 1.e-10, np.abs(
        (new_qg - slice_result_qg) / slice_result_qg), 0.0)
    for k in range(icon_grid.num_levels):
        print("{0:3d} {1:6e} {2:5e} {3:5e} {4:5e} {5:5e} {6:5e} {7:5e}".format(
                k,
                np.abs(
                    new_temperature[:,k] -
                    #exit_savepoint.temperature().ndarray[cell_lower_limit:cell_upper_limit, k]
                    slice_result_t[:,k]
                ).max(),
                np.abs(
                    new_qv[:,k] -
                    #exit_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, k]
                    slice_result_qv[:,k]
                ).max(),
                np.abs(
                    new_qc[:,k] -
                    slice_result_qc[:,k]
                ).max(),
                np.abs(
                    new_qi[:,k] -
                    slice_result_qi[:,k]
                ).max(),
                np.abs(
                    new_qr[:,k] -
                    slice_result_qr[:,k]
                ).max(),
                np.abs(
                    new_qs[:,k] -
                    slice_result_qs[:,k]
                ).max(),
                np.abs(
                    new_qg[:,k] -
                    slice_result_qg[:,k]
                ).max()
        )
        )
    print()
    for k in range(icon_grid.num_levels):
        print("{0:3d} {1:6e} {2:5e} {3:5e} {4:5e} {5:5e} {6:5e} {7:5e}".format(
            k,
            np.abs(relative_error_t[:,k]).max(),
            np.abs(relative_error_qv[:,k]).max(),
            np.abs(relative_error_qc[:,k]).max(),
            np.abs(relative_error_qi[:,k]).max(),
            np.abs(relative_error_qr[:,k]).max(),
            np.abs(relative_error_qs[:,k]).max(),
            np.abs(relative_error_qg[:,k]).max(),
        )
        )

    '''

    assert dallclose(
        new_temperature,
        exit_savepoint.temperature().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )
    assert dallclose(
        new_qv,
        exit_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )
    assert dallclose(
        new_qc,
        exit_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )
    assert dallclose(
        new_qr,
        exit_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )
    assert dallclose(
        new_qi,
        exit_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )
    assert dallclose(
        new_qs,
        exit_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )
    assert dallclose(
        new_qg,
        exit_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit, 0:icon_grid.num_levels]
    )


    print("check")
    print(
        np.abs(graupel_microphysics.temperature_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(graupel_microphysics.qv_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(graupel_microphysics.qc_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(graupel_microphysics.qi_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(graupel_microphysics.qr_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(graupel_microphysics.qs_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(graupel_microphysics.qg_tendency.ndarray[cell_start:cell_end, k_start:k_end]).max()
    )

    print(
        np.abs(exit_savepoint.temperature().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_temperature[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.temperature().ndarray[cell_start:cell_end, k_start:k_end]
        ).max()
    )
    print(
        np.abs(exit_savepoint.qv().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_qv[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.qv().ndarray[cell_start:cell_end,k_start:k_end]
        ).max()
    )
    print(
        np.abs(exit_savepoint.qc().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_qc[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.qc().ndarray[cell_start:cell_end, k_start:k_end]
        ).max()
    )
    print(
        np.abs(exit_savepoint.qi().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_qi[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.qi().ndarray[cell_start:cell_end, k_start:k_end]
        ).max()
    )
    print(
        np.abs(exit_savepoint.qr().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_qr[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.qr().ndarray[cell_start:cell_end, k_start:k_end]
        ).max()
    )
    print(
        np.abs(exit_savepoint.qs().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_qs[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.qs().ndarray[cell_start:cell_end, k_start:k_end]
        ).max()
    )
    print(
        np.abs(exit_savepoint.qg().ndarray[cell_start:cell_end, k_start:k_end]).max(),
        np.abs(
            new_qg[cell_start:cell_end, k_start:k_end] -
            exit_savepoint.qg().ndarray[cell_start:cell_end, k_start:k_end]
        ).max()
    )

    '''






    '''
    graupel_microphysics.run(
        dtime,
        prognostic_state,
        diagnostic_state,
        tracer_state,
    )


    new_temperature = entry_savepoint.temperature().ndarray + graupel_microphysics.temperature_tendency.ndarray * dtime
    new_qv = entry_savepoint.qv().ndarray + graupel_microphysics.qv_tendency.ndarray * dtime
    new_qc = entry_savepoint.qc().ndarray + graupel_microphysics.qc_tendency.ndarray * dtime
    new_qr = entry_savepoint.qr().ndarray + graupel_microphysics.qr_tendency.ndarray * dtime
    new_qi = entry_savepoint.qi().ndarray + graupel_microphysics.qi_tendency.ndarray * dtime
    new_qs = entry_savepoint.qs().ndarray + graupel_microphysics.qs_tendency.ndarray * dtime
    new_qg = entry_savepoint.qg().ndarray + graupel_microphysics.qg_tendency.ndarray * dtime
    # measure differences
    assert dallclose(new_temperature, exit_savepoint.temperature().ndarray)
    assert dallclose(new_qv, exit_savepoint.qv().ndarray)
    assert dallclose(new_qc, exit_savepoint.qc().ndarray)
    assert dallclose(new_qr, exit_savepoint.qr().ndarray)
    assert dallclose(new_qi, exit_savepoint.qi().ndarray)
    assert dallclose(new_qs, exit_savepoint.qs().ndarray)
    assert dallclose(new_qg, exit_savepoint.qg().ndarray)
    assert dallclose(graupel_microphysics.temperature_tendency.ndarray, exit_savepoint.ddt_tend_t().ndarray)
    assert dallclose(graupel_microphysics.qv_tendency.ndarray, exit_savepoint.ddt_tend_qv().ndarray)
    assert dallclose(graupel_microphysics.qc_tendency.ndarray, exit_savepoint.ddt_tend_qc().ndarray)
    assert dallclose(graupel_microphysics.qr_tendency.ndarray, exit_savepoint.ddt_tend_qr().ndarray)
    assert dallclose(graupel_microphysics.qi_tendency.ndarray, exit_savepoint.ddt_tend_qi().ndarray)
    assert dallclose(graupel_microphysics.qs_tendency.ndarray, exit_savepoint.ddt_tend_qs().ndarray)

    assert dallclose(
        graupel_microphysics.ice_precipitation_flux.ndarray[:,icon_grid.num_levels-1],
        exit_savepoint.ice_flux().ndarray
    )
    assert dallclose(
        graupel_microphysics.rain_precipitation_flux.ndarray[:, icon_grid.num_levels - 1],
        exit_savepoint.rain_flux().ndarray
    )
    assert dallclose(
        graupel_microphysics.snow_precipitation_flux.ndarray[:, icon_grid.num_levels - 1],
        exit_savepoint.snow_flux().ndarray
    )
    assert dallclose(
        graupel_microphysics.graupel_precipitation_flux.ndarray[:, icon_grid.num_levels - 1],
        exit_savepoint.graupel_flux().ndarray
    )


    print("Max predict-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(predict_field[item].asnumpy() - ref_data[item]).max(), diff_data[item].max())

    print("Max init-ref difference:")
    for item in mixT_name:
        print(item, ": ", np.abs(ser_field[item].asnumpy() - ref_data[item]).max())


    if data_output:
        with open(base_dir+'analysis_dz_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    f.write(" {0:.20e} ".format(ser_field["ser_graupel_dz"].asnumpy()[i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_ser_rho_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    f.write(" {0:.20e} ".format(ser_data["ser_graupel_rho"][i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_ref_rho_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    f.write(" {0:.20e} ".format(ref_data["ser_graupel_rho"][i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_predict_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write( "{0:7d} {1:7d}".format(i,k))
                    for item in mixT_name:
                        f.write(" {0:.20e} ".format(predict_field[item].asnumpy()[i,k]))
                    f.write("\n")

        with open(base_dir+'analysis_tend_rank' + str(rank) + '.dat', 'w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in ser_tend_name:
                        f.write(" {0:.20e} ".format(tend_field[item].asnumpy()[i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_redundant_rank' + str(rank) + '.dat', 'w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in ser_redundant_name:
                        f.write(" {0:.20e} ".format(redundant_field[item].asnumpy()[i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_velocity_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in velocity_field_name:
                        f.write(" {0:.20e} ".format(velocity_field[item].asnumpy()[i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_ser_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in mixT_name:
                        f.write(" {0:.20e} ".format(ser_data[item][i, k]))
                    f.write("\n")

        with open(base_dir+'analysis_ref_rank'+str(rank)+'.dat','w') as f:
            for i in range(cell_size):
                for k in range(k_size):
                    f.write("{0:7d} {1:7d}".format(i, k))
                    for item in mixT_name:
                        f.write(" {0:.20e} ".format(ref_data[item][i, k]))
                    f.write("\n")

        if tendency_serialization:
            with open(base_dir + 'analysis_ref_tend_rank' + str(rank) + '.dat', 'w') as f:
                for i in range(cell_size):
                    for k in range(k_size):
                        f.write("{0:7d} {1:7d}".format(i, k))
                        for item in ser_tend_name:
                            f.write(" {0:.20e} ".format(tend_data[item][i, k]))
                        f.write("\n")

    '''
