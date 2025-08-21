# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc
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
    ],
)
@pytest.mark.parametrize(
    "date",
    ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"],
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
        liquid_autoconversion_option=graupel.LiquidAutoConversionType.SEIFERT_BEHENG,  # init_savepoint.iautocon(),
        ice_stickeff_min=0.01,
        power_law_coeff_for_ice_mean_fall_speed=1.25,
        exponent_for_density_factor_in_ice_sedimentation=0.30,
        power_law_coeff_for_snow_fall_speed=20.0,
        rain_mu=0.0,
        rain_n0=1.0,
        snow2graupel_riming_coeff=0.5,
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    qnc = entry_savepoint.qnc()

    temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    qv_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    qc_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    qr_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    qi_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    qs_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    qg_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )

    graupel_microphysics.run(
        dtime,
        prognostic_state.rho,
        diagnostic_state.temperature,
        diagnostic_state.pressure,
        tracer_state.qv,
        tracer_state.qc,
        tracer_state.qr,
        tracer_state.qi,
        tracer_state.qs,
        tracer_state.qg,
        qnc,
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qr_tendency,
        qi_tendency,
        qs_tendency,
        qg_tendency,
    )

    new_temperature = (
        entry_savepoint.temperature().ndarray[:, :] + temperature_tendency.ndarray * dtime
    )
    new_qv = entry_savepoint.qv().ndarray[:, :] + qv_tendency.ndarray * dtime
    new_qc = entry_savepoint.qc().ndarray[:, :] + qc_tendency.ndarray * dtime
    new_qr = entry_savepoint.qr().ndarray[:, :] + qr_tendency.ndarray * dtime
    new_qi = entry_savepoint.qi().ndarray[:, :] + qi_tendency.ndarray * dtime
    new_qs = entry_savepoint.qs().ndarray[:, :] + qs_tendency.ndarray * dtime
    new_qg = entry_savepoint.qg().ndarray[:, :] + qg_tendency.ndarray * dtime

    assert test_utils.dallclose(
        new_temperature,
        exit_savepoint.temperature().ndarray[:, :],
    )
    assert test_utils.dallclose(
        new_qv,
        exit_savepoint.qv().ndarray[:, :],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qc,
        exit_savepoint.qc().ndarray[:, :],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qr,
        exit_savepoint.qr().ndarray[:, :],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qi,
        exit_savepoint.qi().ndarray[:, :],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qs,
        exit_savepoint.qs().ndarray[:, :],
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qg,
        exit_savepoint.qg().ndarray[:, :],
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        graupel_microphysics.rain_precipitation_flux.ndarray[:, -1],
        exit_savepoint.rain_flux().ndarray[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.snow_precipitation_flux.ndarray[:, -1],
        exit_savepoint.snow_flux().ndarray[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.graupel_precipitation_flux.ndarray[:, -1],
        exit_savepoint.graupel_flux().ndarray[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.ice_precipitation_flux.ndarray[:, -1],
        exit_savepoint.ice_flux().ndarray[:],
        atol=9.0e-11,
    )
