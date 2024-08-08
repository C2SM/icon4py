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
from icon4py.model.atmosphere.physics.microphysics import saturation_adjustment
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics, tracer_state as tracers
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
def test_saturation_adjustment(
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
    """Test satad aginst a numpy implementaion."""

    entry_microphysics_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    entry_savepoint = data_provider.from_savepoint_weisman_klemp_gscp_satad_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_gscp_satad_exit(date=date)

    config = saturation_adjustment.SaturationAdjustmentConfig(
        tolerance=entry_savepoint.tolerance(),
        max_iter=entry_savepoint.maxiter(),
    )

    saturation_adjustment_granule = saturation_adjustment.SaturationAdjustment(
        config=config,
        grid=icon_grid,
    )

    tracer_state = tracers.TracerState(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qr=None,
        qi=None,
        qs=None,
        qg=None,
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
        pressure=None,
        pressure_ifc=None,
        u=None,
        v=None,
    )

    dtime = entry_microphysics_savepoint.dt_microphysics()
    saturation_adjustment_granule.run(
        dtime=dtime,
        prognostic_state=prognostic_state,
        diagnostic_state=diagnostic_state,
        tracer_state=tracer_state,
    )

    updated_qv = tracer_state.qv.ndarray + saturation_adjustment_granule.qv_tendency.ndarray * dtime
    updated_qc = tracer_state.qc.ndarray + saturation_adjustment_granule.qc_tendency.ndarray * dtime
    updated_temperature = diagnostic_state.temperature.ndarray + saturation_adjustment_granule.temperature_tendency.ndarray * dtime

    assert dallclose(
        updated_qv,
        exit_savepoint.qv().ndarray,
        atol=1.e-13,
    )
    assert dallclose(
        updated_qc,
        exit_savepoint.qc().ndarray,
        atol = 1.e-13,
    )
    assert dallclose(
        updated_temperature,
        exit_savepoint.temperature().ndarray,
        atol = 1.e-13,
    )
