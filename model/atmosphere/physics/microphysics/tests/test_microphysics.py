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

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.physics.microphysics import (
    saturation_adjustment,
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.parametrize(
    "experiment, model_top_height,, damping_height, stretch_factor, date",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "48"),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "52"),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85, "56"),
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
    assert dallclose(graupel_microphysics.ccs[0], init_savepoint.ccsrim(), atol=1.0e-8)
    assert dallclose(graupel_microphysics.ccs[1], init_savepoint.ccsagg(), atol=1.0e-8)
    assert dallclose(graupel.icon_graupel_params.ccsaxp, init_savepoint.ccsaxp())
    assert dallclose(graupel.icon_graupel_params.ccsdep, init_savepoint.ccsdep(), atol=1.0e-7)
    assert dallclose(graupel.icon_graupel_params.ccsdxp, init_savepoint.ccsdxp())
    assert dallclose(graupel_microphysics.ccs[2], init_savepoint.ccsvel(), atol=1.0e-8)
    assert dallclose(graupel.icon_graupel_params.ccsvxp, init_savepoint.ccsvxp())
    assert dallclose(graupel.icon_graupel_params.ccslam, init_savepoint.ccslam(), atol=1.0e-10)
    assert dallclose(graupel.icon_graupel_params.ccslxp, init_savepoint.ccslxp())
    assert dallclose(graupel.icon_graupel_params.ccshi1, init_savepoint.ccshi1())
    assert dallclose(graupel.icon_graupel_params.ccdvtp, init_savepoint.ccdvtp())
    assert dallclose(graupel.icon_graupel_params.ccidep, init_savepoint.ccidep())
    assert dallclose(graupel_microphysics.rain_vel_coef[2], init_savepoint.cevxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[3], init_savepoint.cev(), atol=1.0e-10)
    assert dallclose(graupel_microphysics.rain_vel_coef[4], init_savepoint.bevxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[5], init_savepoint.bev())
    assert dallclose(graupel_microphysics.rain_vel_coef[0], init_savepoint.vzxp())
    assert dallclose(graupel_microphysics.rain_vel_coef[1], init_savepoint.vz0r(), atol=1.0e-10)

    # TODO (Chia Rui): remove this slicing process, which finds the column with maximum tendency, when either the scan operator can be run on the gtfn backend or running on embedded backend is faster
    max_index = np.unravel_index(
        np.abs(tracer_state.qv.ndarray - exit_savepoint.qv().ndarray).argmax(),
        exit_savepoint.qv().ndarray.shape,
    )
    cell_lower_limit = max_index[0] - 300  # - 277
    cell_upper_limit = max_index[0] + 300  # - 276
    cell_size = cell_upper_limit - cell_lower_limit
    vertical_size = icon_grid.num_levels

    slice_t = gtx.as_field(
        (CellDim, KDim), diagnostic_state.temperature.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_pres = gtx.as_field(
        (CellDim, KDim), diagnostic_state.pressure.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_rho = gtx.as_field(
        (CellDim, KDim), prognostic_state.rho.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qv = gtx.as_field(
        (CellDim, KDim), tracer_state.qv.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qc = gtx.as_field(
        (CellDim, KDim), tracer_state.qc.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qi = gtx.as_field(
        (CellDim, KDim), tracer_state.qi.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qr = gtx.as_field(
        (CellDim, KDim), tracer_state.qr.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qs = gtx.as_field(
        (CellDim, KDim), tracer_state.qs.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qg = gtx.as_field(
        (CellDim, KDim), tracer_state.qg.ndarray[cell_lower_limit:cell_upper_limit, :]
    )

    qnc_2d = entry_savepoint.qnc().ndarray
    qnc = gtx.as_field(
        (CellDim, KDim), np.repeat(np.expand_dims(qnc_2d, axis=-1), icon_grid.num_levels, axis=1)
    )
    slice_qnc = gtx.as_field((CellDim, KDim), qnc.ndarray[cell_lower_limit:cell_upper_limit, :])
    slice_ddqz_z_full = gtx.as_field(
        (CellDim, KDim),
        graupel_microphysics.metric_state.ddqz_z_full.ndarray[cell_lower_limit:cell_upper_limit, :],
    )

    slice_t_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qv_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qc_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qi_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qr_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qs_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_qg_t = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))

    slice_rhoqrv_old_kup = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rhoqsv_old_kup = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rhoqgv_old_kup = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rhoqiv_old_kup = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_vnew_r = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_s = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_g = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_vnew_i = gtx.as_field((CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float))
    slice_rain_precipitation_flux = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_snow_precipitation_flux = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_graupel_precipitation_flux = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_ice_precipitation_flux = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_total_precipitation_flux = gtx.as_field(
        (CellDim, KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )

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
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(cell_size),
        vertical_start=gtx.int32(0),
        vertical_end=vertical_size,
        offset_provider={},
    )
    new_temperature = (
        entry_savepoint.temperature().ndarray[
            cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
        ]
        + slice_t_t.ndarray * dtime
    )
    new_qv = (
        entry_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels]
        + slice_qv_t.ndarray * dtime
    )
    new_qc = (
        entry_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels]
        + slice_qc_t.ndarray * dtime
    )
    new_qr = (
        entry_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels]
        + slice_qr_t.ndarray * dtime
    )
    new_qi = (
        entry_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels]
        + slice_qi_t.ndarray * dtime
    )
    new_qs = (
        entry_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels]
        + slice_qs_t.ndarray * dtime
    )
    new_qg = (
        entry_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels]
        + slice_qg_t.ndarray * dtime
    )

    graupel.icon_graupel_flux_above_ground(
        graupel_microphysics.config.do_latent_heat_nudging,
        dtime,
        slice_rho,
        slice_qr,
        slice_qs,
        slice_qg,
        slice_qi,
        slice_qr_t,
        slice_qs_t,
        slice_qg_t,
        slice_qi_t,
        slice_rhoqrv_old_kup,
        slice_rhoqsv_old_kup,
        slice_rhoqgv_old_kup,
        slice_rhoqiv_old_kup,
        slice_vnew_r,
        slice_vnew_s,
        slice_vnew_g,
        slice_vnew_i,
        slice_rain_precipitation_flux,
        slice_snow_precipitation_flux,
        slice_graupel_precipitation_flux,
        slice_ice_precipitation_flux,
        slice_total_precipitation_flux,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(cell_size),
        vertical_start=gtx.int32(vertical_params.kstart_moist),
        vertical_end=vertical_size - gtx.int32(1),
        offset_provider={},
    )

    graupel.icon_graupel_flux_ground(
        graupel_microphysics.config.do_latent_heat_nudging,
        dtime,
        slice_rho,
        slice_qr,
        slice_qs,
        slice_qg,
        slice_qi,
        slice_qr_t,
        slice_qs_t,
        slice_qg_t,
        slice_qi_t,
        slice_rhoqrv_old_kup,
        slice_rhoqsv_old_kup,
        slice_rhoqgv_old_kup,
        slice_rhoqiv_old_kup,
        slice_vnew_r,
        slice_vnew_s,
        slice_vnew_g,
        slice_vnew_i,
        slice_rain_precipitation_flux,
        slice_snow_precipitation_flux,
        slice_graupel_precipitation_flux,
        slice_ice_precipitation_flux,
        slice_total_precipitation_flux,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(cell_size),
        vertical_start=vertical_size - gtx.int32(1),
        vertical_end=vertical_size,
        offset_provider={},
    )

    assert dallclose(
        new_temperature,
        exit_savepoint.temperature().ndarray[
            cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
        ],
    )
    assert dallclose(
        new_qv,
        exit_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert dallclose(
        new_qc,
        exit_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert dallclose(
        new_qr,
        exit_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert dallclose(
        new_qi,
        exit_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert dallclose(
        new_qs,
        exit_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert dallclose(
        new_qg,
        exit_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )

    assert dallclose(
        slice_rain_precipitation_flux.ndarray[:, -1],
        exit_savepoint.rain_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )
    assert dallclose(
        slice_snow_precipitation_flux.ndarray[:, -1],
        exit_savepoint.snow_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )

    assert dallclose(
        slice_graupel_precipitation_flux.ndarray[:, -1],
        exit_savepoint.graupel_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )

    assert dallclose(
        slice_ice_precipitation_flux.ndarray[:, -1],
        exit_savepoint.ice_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )
