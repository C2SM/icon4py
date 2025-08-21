# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    test_utils,
)

from ..fixtures import *  # noqa: F403


@pytest.mark.parametrize(
    "experiment, model_top_height, damping_height, stretch_factor",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85),
        # (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85),
        # (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85),
    ],
)
@pytest.mark.parametrize(
    "date",
    [
        "2008-09-01T01:59:48.000",
    ],  # , "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"
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
    backend,
):
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
    )

    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_exit(date=date)

    dtime = entry_savepoint.dtime()

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
        virtual_temperature=None,
        pressure=entry_savepoint.pressure(),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=graupel.LiquidAutoConversion.KESSLER,  # init_savepoint.iautocon(),
        ice_stickeff_min=0.075,  # init_savepoint.ceff_min(),
        power_law_coeff_for_ice_mean_fall_speed=1.25,  # init_savepoint.vz0i(),
        exponent_for_density_factor_in_ice_sedimentation=0.33,  # init_savepoint.icesedi_exp(),
        power_law_coeff_for_snow_fall_speed=20.0,  # init_savepoint.v0snow(),
        rain_mu=0.0,  # init_savepoint.mu_rain(),
        rain_n0=1.0,  # init_savepoint.rain_n0_factor(),
        snow2graupel_riming_coeff=0.5,
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    # assert test_utils.dallclose(graupel.icon_graupel_params.qmin, init_savepoint.qmin())
    # assert test_utils.dallclose(
    #     graupel.icon_graupel_params.power_law_coeff_for_snow_mD_relation, init_savepoint.ams()
    # )
    # assert test_utils.dallclose(graupel_microphysics.ccs[0], init_savepoint.ccsrim(), atol=1.0e-8)
    # assert test_utils.dallclose(graupel_microphysics.ccs[1], init_savepoint.ccsagg(), atol=1.0e-8)
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccsaxp, init_savepoint.ccsaxp())
    # assert test_utils.dallclose(
    #     graupel.icon_graupel_params.ccsdep, init_savepoint.ccsdep(), atol=1.0e-7
    # )
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccsdxp, init_savepoint.ccsdxp())
    # assert test_utils.dallclose(graupel_microphysics.ccs[2], init_savepoint.ccsvel(), atol=1.0e-8)
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccsvxp, init_savepoint.ccsvxp())
    # assert test_utils.dallclose(
    #     graupel.icon_graupel_params.ccslam, init_savepoint.ccslam(), atol=1.0e-10
    # )
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccslxp, init_savepoint.ccslxp())
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccshi1, init_savepoint.ccshi1())
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccdvtp, init_savepoint.ccdvtp())
    # assert test_utils.dallclose(graupel.icon_graupel_params.ccidep, init_savepoint.ccidep())
    # assert test_utils.dallclose(graupel_microphysics.rain_vel_coef[0], init_savepoint.vzxp())
    # assert test_utils.dallclose(
    #     graupel_microphysics.rain_vel_coef[1], init_savepoint.vz0r(), atol=1.0e-10
    # )
    # assert test_utils.dallclose(graupel_microphysics.rain_vel_coef[2], init_savepoint.cevxp())
    # assert test_utils.dallclose(
    #     graupel_microphysics.rain_vel_coef[3], init_savepoint.cev(), atol=1.0e-10
    # )
    # assert test_utils.dallclose(graupel_microphysics.rain_vel_coef[4], init_savepoint.bevxp())
    # assert test_utils.dallclose(graupel_microphysics.rain_vel_coef[5], init_savepoint.bev())

    # TODO (Chia Rui): remove this slicing process, which finds the column with maximum tendency, when either the scan operator can be run on the gtfn backend or running on embedded backend is faster
    max_index = np.unravel_index(
        np.abs(tracer_state.qv.asnumpy() - exit_savepoint.qv().asnumpy()).argmax(),
        exit_savepoint.qv().asnumpy().shape,
    )
    # print()
    # print("DEBUG", max_index, icon_grid.num_levels, diagnostic_state.temperature.ndarray.shape)
    # print("DEBUG", np.abs(tracer_state.qv.asnumpy() - exit_savepoint.qv().asnumpy()).max())
    # print("DEBUG", tracer_state.qv.asnumpy().max(), tracer_state.qv.asnumpy().min())
    # print("DEBUG", exit_savepoint.qv().asnumpy().max(), exit_savepoint.qv().asnumpy().min())
    # return
    cell_lower_limit = 0  # max_index[0] - 300
    cell_upper_limit = exit_savepoint.qv().asnumpy().shape[0]  # max_index[0] + 300
    cell_size = cell_upper_limit - cell_lower_limit
    vertical_size = icon_grid.num_levels

    slice_t = gtx.as_field(
        (dims.CellDim, dims.KDim),
        diagnostic_state.temperature.ndarray[cell_lower_limit:cell_upper_limit, :],
    )
    slice_pres = gtx.as_field(
        (dims.CellDim, dims.KDim),
        diagnostic_state.pressure.ndarray[cell_lower_limit:cell_upper_limit, :],
    )
    slice_rho = gtx.as_field(
        (dims.CellDim, dims.KDim),
        prognostic_state.rho.ndarray[cell_lower_limit:cell_upper_limit, :],
    )
    slice_qv = gtx.as_field(
        (dims.CellDim, dims.KDim), tracer_state.qv.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qc = gtx.as_field(
        (dims.CellDim, dims.KDim), tracer_state.qc.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qi = gtx.as_field(
        (dims.CellDim, dims.KDim), tracer_state.qi.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qr = gtx.as_field(
        (dims.CellDim, dims.KDim), tracer_state.qr.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qs = gtx.as_field(
        (dims.CellDim, dims.KDim), tracer_state.qs.ndarray[cell_lower_limit:cell_upper_limit, :]
    )
    slice_qg = gtx.as_field(
        (dims.CellDim, dims.KDim), tracer_state.qg.ndarray[cell_lower_limit:cell_upper_limit, :]
    )

    qnc_2d = entry_savepoint.qnc().ndarray
    slice_qnc = gtx.as_field((dims.CellDim,), qnc_2d[cell_lower_limit:cell_upper_limit])
    slice_ddqz_z_full = gtx.as_field(
        (dims.CellDim, dims.KDim),
        graupel_microphysics.metric_state.ddqz_z_full.ndarray[cell_lower_limit:cell_upper_limit, :],
    )

    slice_t_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_qv_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_qc_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_qi_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_qr_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_qs_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_qg_t = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )

    slice_rhoqrv_old_kup = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rhoqsv_old_kup = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rhoqgv_old_kup = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rhoqiv_old_kup = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_vnew_r = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_vnew_s = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_vnew_g = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_vnew_i = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_rain_precipitation_flux = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_snow_precipitation_flux = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_graupel_precipitation_flux = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_ice_precipitation_flux = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )
    slice_total_precipitation_flux = gtx.as_field(
        (dims.CellDim, dims.KDim), np.zeros((cell_size, vertical_size), dtype=float)
    )

    graupel.icon_graupel(
        vertical_params.kstart_moist,
        gtx.int32(icon_grid.num_levels) - gtx.int32(1),
        graupel_microphysics.config.liquid_autoconversion_option,
        graupel_microphysics.config.snow_intercept_option,
        graupel_microphysics.config.is_isochoric,
        graupel_microphysics.config.use_constant_water_heat_capacity,
        graupel_microphysics.config.ice_stickeff_min,
        graupel_microphysics.config.snow2graupel_riming_coeff,
        graupel_microphysics.config.power_law_coeff_for_ice_mean_fall_speed,
        graupel_microphysics.config.exponent_for_density_factor_in_ice_sedimentation,
        graupel_microphysics.config.power_law_coeff_for_snow_fall_speed,
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
        vertical_end=gtx.int32(vertical_size - 1),
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
        vertical_start=gtx.int32(vertical_size - 1),
        vertical_end=gtx.int32(vertical_size),
        offset_provider={},
    )

    print()
    print(
        np.abs(
            new_temperature
            - exit_savepoint.temperature().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            new_qv
            - exit_savepoint.qv().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            new_qc
            - exit_savepoint.qc().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            new_qi
            - exit_savepoint.qi().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            new_qr
            - exit_savepoint.qr().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            new_qs
            - exit_savepoint.qs().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            new_qg
            - exit_savepoint.qg().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print()
    print(
        np.abs(
            slice_t.ndarray
            - exit_savepoint.temperature().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            slice_qv.ndarray
            - exit_savepoint.qv().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            slice_qc.ndarray
            - exit_savepoint.qc().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            slice_qi.ndarray
            - exit_savepoint.qi().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            slice_qr.ndarray
            - exit_savepoint.qr().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            slice_qs.ndarray
            - exit_savepoint.qs().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    print(
        np.abs(
            slice_qg.ndarray
            - exit_savepoint.qg().ndarray[
                cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
            ]
        ).max()
    )
    assert test_utils.dallclose(
        new_temperature,
        exit_savepoint.temperature().ndarray[
            cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels
        ],
    )
    assert test_utils.dallclose(
        new_qv,
        exit_savepoint.qv().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qc,
        exit_savepoint.qc().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qr,
        exit_savepoint.qr().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qi,
        exit_savepoint.qi().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qs,
        exit_savepoint.qs().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qg,
        exit_savepoint.qg().ndarray[cell_lower_limit:cell_upper_limit, 0 : icon_grid.num_levels],
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        slice_rain_precipitation_flux.ndarray[:, -1],
        exit_savepoint.rain_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        slice_snow_precipitation_flux.ndarray[:, -1],
        exit_savepoint.snow_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        slice_graupel_precipitation_flux.ndarray[:, -1],
        exit_savepoint.graupel_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        slice_ice_precipitation_flux.ndarray[:, -1],
        exit_savepoint.ice_flux().ndarray[cell_lower_limit:cell_upper_limit],
        atol=1.0e-12,
    )
