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


import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import saturation_adjustment
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import dallclose, zero_field


@pytest.mark.parametrize(
    "experiment, model_top_height, damping_height, stretch_factor, date",
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
    entry_microphysics_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(
        date=date
    )
    gscp_satad_entry_savepoint = data_provider.from_savepoint_weisman_klemp_gscp_satad_entry(
        date=date
    )
    gscp_satad_exit_savepoint = data_provider.from_savepoint_weisman_klemp_gscp_satad_exit(
        date=date
    )

    gscp_satad_config = saturation_adjustment.SaturationAdjustmentConfig(
        tolerance=gscp_satad_entry_savepoint.tolerance(),
        max_iter=gscp_satad_entry_savepoint.maxiter(),
        diagnose_variables_from_new_temperature=False,
    )

    vertical_config = v_grid.VerticalGridConfig(icon_grid.num_levels)
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )

    metric_state = saturation_adjustment.MetricStateSaturationAdjustment(
        ddqz_z_full=metrics_savepoint.ddqz_z_full()
    )

    gscp_saturation_adjustment_granule = saturation_adjustment.SaturationAdjustment(
        config=gscp_satad_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
    )

    tracer_state = tracers.TracerState(
        qv=gscp_satad_entry_savepoint.qv(),
        qc=gscp_satad_entry_savepoint.qc(),
        qr=None,
        qi=None,
        qs=None,
        qg=None,
    )
    prognostic_state = prognostics.PrognosticState(
        rho=gscp_satad_entry_savepoint.rho(),
        vn=None,
        w=None,
        exner=None,
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=gscp_satad_entry_savepoint.temperature(),
        virtual_temperature=None,
        pressure=zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure_ifc=zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}
        ),
        u=None,
        v=None,
    )

    dtime = entry_microphysics_savepoint.dt_microphysics()
    gscp_saturation_adjustment_granule.run(
        dtime=dtime,
        prognostic_state=prognostic_state,
        diagnostic_state=diagnostic_state,
        tracer_state=tracer_state,
    )

    updated_qv = (
        tracer_state.qv.ndarray + gscp_saturation_adjustment_granule.qv_tendency.ndarray * dtime
    )
    updated_qc = (
        tracer_state.qc.ndarray + gscp_saturation_adjustment_granule.qc_tendency.ndarray * dtime
    )
    updated_temperature = (
        diagnostic_state.temperature.ndarray
        + gscp_saturation_adjustment_granule.temperature_tendency.ndarray * dtime
    )

    assert dallclose(
        updated_qv,
        gscp_satad_exit_savepoint.qv().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_qc,
        gscp_satad_exit_savepoint.qc().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_temperature,
        gscp_satad_exit_savepoint.temperature().ndarray,
        atol=1.0e-13,
    )

    nwp_interface_satad_entry_savepoint = (
        data_provider.from_savepoint_weisman_klemp_interface_satad_entry(date=date)
    )
    nwp_interface_satad_exit_savepoint = (
        data_provider.from_savepoint_weisman_klemp_interface_satad_exit(date=date)
    )

    nwp_interface_satad_diag_entry_savepint = (
        data_provider.from_savepoint_weisman_klemp_interface_diag_after_satad_entry(date=date)
    )
    nwp_interface_satad_diag_exit_savepint = (
        data_provider.from_savepoint_weisman_klemp_interface_diag_after_satad_exit(date=date)
    )

    nwp_interface_config = saturation_adjustment.SaturationAdjustmentConfig(
        tolerance=nwp_interface_satad_entry_savepoint.tolerance(),
        max_iter=nwp_interface_satad_entry_savepoint.maxiter(),
        diagnose_variables_from_new_temperature=True,
    )

    nwp_interface_saturation_adjustment_granule = saturation_adjustment.SaturationAdjustment(
        config=nwp_interface_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
    )

    tracer_state = tracers.TracerState(
        qv=nwp_interface_satad_entry_savepoint.qv(),
        qc=nwp_interface_satad_entry_savepoint.qc(),
        qr=nwp_interface_satad_diag_entry_savepint.qr(),
        qi=nwp_interface_satad_diag_entry_savepint.qi(),
        qs=nwp_interface_satad_diag_entry_savepint.qs(),
        qg=nwp_interface_satad_diag_entry_savepint.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=nwp_interface_satad_entry_savepoint.rho(),
        vn=None,
        w=None,
        exner=nwp_interface_satad_diag_entry_savepint.exner(),
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=nwp_interface_satad_entry_savepoint.temperature(),
        virtual_temperature=nwp_interface_satad_diag_entry_savepint.virtual_temperature(),
        pressure=zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float),
        pressure_ifc=zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}
        ),
        u=None,
        v=None,
    )

    nwp_interface_saturation_adjustment_granule.run(
        dtime=dtime,
        prognostic_state=prognostic_state,
        diagnostic_state=diagnostic_state,
        tracer_state=tracer_state,
    )

    updated_qv = (
        tracer_state.qv.ndarray
        + nwp_interface_saturation_adjustment_granule.qv_tendency.ndarray * dtime
    )
    updated_qc = (
        tracer_state.qc.ndarray
        + nwp_interface_saturation_adjustment_granule.qc_tendency.ndarray * dtime
    )
    updated_temperature = (
        diagnostic_state.temperature.ndarray
        + nwp_interface_saturation_adjustment_granule.temperature_tendency.ndarray * dtime
    )
    updated_virtual_temperature = (
        diagnostic_state.virtual_temperature.ndarray
        + nwp_interface_saturation_adjustment_granule.virtual_temperature_tendency.ndarray * dtime
    )
    updated_exner = (
        prognostic_state.exner.ndarray
        + nwp_interface_saturation_adjustment_granule.exner_tendency.ndarray * dtime
    )
    updated_pressure = (
        diagnostic_state.pressure.ndarray
        + nwp_interface_saturation_adjustment_granule.pressure_tendency.ndarray * dtime
    )
    updated_pressure_ifc = (
        diagnostic_state.pressure_ifc.ndarray
        + nwp_interface_saturation_adjustment_granule.pressure_ifc_tendency.ndarray * dtime
    )

    assert dallclose(
        updated_qv,
        nwp_interface_satad_exit_savepoint.qv().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_qc,
        nwp_interface_satad_exit_savepoint.qc().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_temperature,
        nwp_interface_satad_exit_savepoint.temperature().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_virtual_temperature,
        nwp_interface_satad_diag_exit_savepint.virtual_temperature().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_exner,
        nwp_interface_satad_diag_exit_savepint.exner().ndarray,
        atol=1.0e-13,
    )

    assert dallclose(
        updated_pressure,
        nwp_interface_satad_diag_exit_savepint.pressure().ndarray,
        atol=1.0e-13,
    )
    assert dallclose(
        updated_pressure_ifc,
        nwp_interface_satad_diag_exit_savepint.pressure_ifc().ndarray,
        atol=1.0e-13,
    )
